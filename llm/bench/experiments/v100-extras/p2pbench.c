/* p2pbench.c — measure peer (NVLink) vs PCIe host-staged copy bandwidth.
 *
 * Built with plain gcc; no CUDA toolkit headers or nvcc are needed because
 * the CUDA *runtime* API is loaded with dlopen() from the same libcudart.so.12
 * the Strata engine uses (pass --cudart, the orchestrator does).  This is
 * deliberate: the node's only nvcc is CUDA 13.4, which no longer emits sm_70
 * code, and the vLLM venv's torch is not a trustworthy measurement device.
 * Using the engine's own runtime also means the numbers describe exactly the
 * API and library the engine calls.
 *
 * Tests (payload 256 MiB unless --mib says otherwise):
 *   local-d2d        GPU0 -> GPU0                 (HBM sanity)
 *   peer-0->1        cudaMemcpyPeerAsync          (same NUMA pair)
 *   peer-0->2        cudaMemcpyPeerAsync          (cross NUMA pair)
 *   default-0->1     cudaMemcpyAsync(Default)     (what a generic D2D call does
 *                                                  once peer access is on)
 *   d2h-0 / h2d-0    GPU0 <-> pinned host         (PCIe, one direction)
 *   staged-0->1      GPU0 -> host -> GPU1         (no-P2P fallback cost)
 *
 * Peer bandwidth is timed with CUDA events around N async copies; the host
 * round trip is timed with CLOCK_MONOTONIC around N synchronised pairs.
 * Effective GB/s for the staged route counts both directions.
 *
 * usage: p2pbench [--mib 256] [--iters 20] [--json out.json]
 *                 [--cudart /path/to/libcudart.so.12]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef int cudaMemcpyKind;

#define CUDA_SUCCESS 0
#define H2D 1
#define D2H 2
#define D2D 3
#define DEFAULT 4
#define HOSTALLOC_PORTABLE 0x01

static cudaError_t (*cudaGetDeviceCount)(int *);
static cudaError_t (*cudaGetDeviceProperties)(void *, int); /* unused */
static cudaError_t (*cudaSetDevice)(int);
static cudaError_t (*cudaMalloc)(void **, size_t);
static cudaError_t (*cudaFree)(void *);
static cudaError_t (*cudaHostAlloc)(void **, size_t, unsigned);
static cudaError_t (*cudaFreeHost)(void *);
static cudaError_t (*cudaDeviceCanAccessPeer)(int *, int, int);
static cudaError_t (*cudaDeviceEnablePeerAccess)(int, unsigned);
static cudaError_t (*cudaMemcpyPeerAsync)(void *, int, const void *, int,
                                          size_t, cudaStream_t);
static cudaError_t (*cudaMemcpyAsync)(void *, const void *, size_t,
                                      cudaMemcpyKind, cudaStream_t);
static cudaError_t (*cudaStreamCreate)(cudaStream_t *);
static cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
static cudaError_t (*cudaEventCreate)(cudaEvent_t *);
static cudaError_t (*cudaEventRecord)(cudaEvent_t, cudaStream_t);
static cudaError_t (*cudaEventSynchronize)(cudaEvent_t);
static cudaError_t (*cudaEventElapsedTime)(float *, cudaEvent_t, cudaEvent_t);
static cudaError_t (*cudaDeviceSynchronize)(void);
static cudaError_t (*cudaGetErrorString_)(cudaError_t);
static const char *(*cudaGetErrorString)(cudaError_t);

static void *lib;
static int failures;
static int json_count;

static void *sym(const char *name) {
  void *p = dlsym(lib, name);
  if (!p) {
    fprintf(stderr, "missing runtime symbol %s\n", name);
    exit(2);
  }
  return p;
}

static void check(cudaError_t rc, const char *what) {
  if (rc != CUDA_SUCCESS) {
    fprintf(stderr, "%s failed: %d (%s)\n", what, rc,
            cudaGetErrorString ? cudaGetErrorString(rc) : "?");
    failures++;
  }
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

struct copy_arg { void *dst; int dst_dev; const void *src; int src_dev;
                  size_t bytes; cudaStream_t s; cudaMemcpyKind kind; };

static void body_peer(void *p) {
  struct copy_arg *a = p;
  cudaMemcpyPeerAsync(a->dst, a->dst_dev, a->src, a->src_dev, a->bytes, a->s);
}
static void body_async(void *p) {
  struct copy_arg *a = p;
  cudaMemcpyAsync(a->dst, a->src, a->bytes, a->kind, a->s);
}

static double event_gbps(cudaStream_t s, size_t bytes, int iters, int multiply,
                         void (*body)(void *), void *arg) {
  cudaEvent_t a, b;
  float ms = 0.0f;
  check(cudaEventCreate(&a), "cudaEventCreate");
  check(cudaEventCreate(&b), "cudaEventCreate");
  check(cudaEventRecord(a, s), "cudaEventRecord");
  for (int i = 0; i < iters; i++) body(arg);
  check(cudaEventRecord(b, s), "cudaEventRecord");
  check(cudaEventSynchronize(b), "cudaEventSynchronize");
  check(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
  if (ms <= 0.0f) return -1.0;
  return (double)bytes * iters * multiply / (ms / 1000.0) / 1e9;
}

static void report(FILE *json, const char *test, double gbps, size_t bytes,
                   int iters, int multiply, const char *note) {
  printf("%-12s %8.2f GB/s  (%zu MiB x %d%s)%s%s\n", test, gbps,
         bytes >> 20, iters, multiply == 2 ? ", both directions" : "",
         note ? "  " : "", note ? note : "");
  if (json) {
    fprintf(json, "%s{\"test\": \"%s\", \"gbps\": %.3f, \"bytes\": %zu, "
                  "\"iters\": %d, \"directions\": %d}",
            json_count++ ? ", " : "", test, gbps, bytes, iters, multiply);
  }
}

int main(int argc, char **argv) {
  size_t mib = 256;
  int iters = 20;
  const char *json_path = NULL;
  const char *cudart = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mib") && i + 1 < argc) mib = strtoul(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else if (!strcmp(argv[i], "--cudart") && i + 1 < argc) cudart = argv[++i];
    else { fprintf(stderr, "usage: %s [--mib N] [--iters N] [--json FILE] "
                           "[--cudart PATH]\n", argv[0]); return 2; }
  }
  const size_t bytes = mib << 20;

  const char *candidates[4] = {
    cudart,
    "libcudart.so.12",
    "/work/ffa/holecem/strata-engine/components/engine/0.1.50-linux-x86_64-cuda-cuda12/libcudart.so.12",
    NULL,
  };
  for (int i = 0; candidates[i] && !lib; i++) lib = dlopen(candidates[i], RTLD_NOW);
  if (!lib) { fprintf(stderr, "cannot load libcudart: %s\n", dlerror()); return 2; }

  cudaGetDeviceCount = (cudaError_t (*)(int *))sym("cudaGetDeviceCount");
  cudaSetDevice = (cudaError_t (*)(int))sym("cudaSetDevice");
  cudaMalloc = (cudaError_t (*)(void **, size_t))sym("cudaMalloc");
  cudaFree = (cudaError_t (*)(void *))sym("cudaFree");
  cudaHostAlloc = (cudaError_t (*)(void **, size_t, unsigned))sym("cudaHostAlloc");
  cudaFreeHost = (cudaError_t (*)(void *))sym("cudaFreeHost");
  cudaDeviceCanAccessPeer = (cudaError_t (*)(int *, int, int))sym("cudaDeviceCanAccessPeer");
  cudaDeviceEnablePeerAccess = (cudaError_t (*)(int, unsigned))sym("cudaDeviceEnablePeerAccess");
  cudaMemcpyPeerAsync = (cudaError_t (*)(void *, int, const void *, int, size_t,
                                         cudaStream_t))sym("cudaMemcpyPeerAsync");
  cudaMemcpyAsync = (cudaError_t (*)(void *, const void *, size_t,
                                     cudaMemcpyKind, cudaStream_t))sym("cudaMemcpyAsync");
  cudaStreamCreate = (cudaError_t (*)(cudaStream_t *))sym("cudaStreamCreate");
  cudaStreamSynchronize = (cudaError_t (*)(cudaStream_t))sym("cudaStreamSynchronize");
  cudaEventCreate = (cudaError_t (*)(cudaEvent_t *))sym("cudaEventCreate");
  cudaEventRecord = (cudaError_t (*)(cudaEvent_t, cudaStream_t))sym("cudaEventRecord");
  cudaEventSynchronize = (cudaError_t (*)(cudaEvent_t))sym("cudaEventSynchronize");
  cudaEventElapsedTime = (cudaError_t (*)(float *, cudaEvent_t, cudaEvent_t))sym("cudaEventElapsedTime");
  cudaDeviceSynchronize = (cudaError_t (*)(void))sym("cudaDeviceSynchronize");
  cudaGetErrorString = (const char *(*)(cudaError_t))dlsym(lib, "cudaGetErrorString");

  int n = 0;
  check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
  printf("runtime devices=%d\n", n);
  if (n < 3) { fprintf(stderr, "need 3 GPUs, found %d\n", n); return 2; }

  void *buf[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4 && i < n; i++) {
    check(cudaSetDevice(i), "cudaSetDevice");
    check(cudaMalloc(&buf[i], bytes), "cudaMalloc");
  }
  if (failures) { fprintf(stderr, "device allocation failed; aborting\n"); return 2; }

  int can01 = -1, can02 = -1;
  cudaDeviceCanAccessPeer(&can01, 0, 1);
  cudaDeviceCanAccessPeer(&can02, 0, 2);
  printf("canAccessPeer 0->1=%d 0->2=%d\n", can01, can02);

  cudaError_t e01, e02, e10;
  check(cudaSetDevice(0), "cudaSetDevice");
  e01 = cudaDeviceEnablePeerAccess(1, 0);
  e02 = cudaDeviceEnablePeerAccess(2, 0);
  check(cudaSetDevice(1), "cudaSetDevice");
  e10 = cudaDeviceEnablePeerAccess(0, 0);
  printf("enablePeerAccess 0->1=%d 0->2=%d 1->0=%d (0 = success, "
         "704 = already enabled)\n", e01, e02, e10);

  void *host = NULL;
  check(cudaSetDevice(0), "cudaSetDevice");
  check(cudaHostAlloc(&host, bytes, HOSTALLOC_PORTABLE), "cudaHostAlloc");
  if (!host) { fprintf(stderr, "pinned host allocation failed\n"); return 2; }
  memset(host, 0x5a, bytes);  /* fault the pinned pages in before timing */

  cudaStream_t s0, s1;
  check(cudaSetDevice(0), "cudaSetDevice");
  check(cudaStreamCreate(&s0), "cudaStreamCreate");
  check(cudaSetDevice(1), "cudaSetDevice");
  check(cudaStreamCreate(&s1), "cudaStreamCreate");
  if (failures) { fprintf(stderr, "stream creation failed; aborting\n"); return 2; }

  FILE *json = json_path ? fopen(json_path, "w") : NULL;
  if (json) fprintf(json, "[");
  struct copy_arg a;

  /* warm-up (first touch of both allocations) */
  check(cudaSetDevice(0), "cudaSetDevice");
  a = (struct copy_arg){buf[1], 1, buf[0], 0, bytes, s0, 0};
  body_peer(&a);
  check(cudaStreamSynchronize(s0), "cudaStreamSynchronize");

  a = (struct copy_arg){buf[0], 0, buf[0], 0, bytes, s0, D2D};
  report(json, "local-d2d", event_gbps(s0, bytes, iters, 1, body_async, &a),
         bytes, iters, 1, "HBM sanity");

  a = (struct copy_arg){buf[1], 1, buf[0], 0, bytes, s0, 0};
  report(json, "peer-0->1", event_gbps(s0, bytes, iters, 1, body_peer, &a),
         bytes, iters, 1, "same NUMA pair");

  a = (struct copy_arg){buf[2], 2, buf[0], 0, bytes, s0, 0};
  report(json, "peer-0->2", event_gbps(s0, bytes, iters, 1, body_peer, &a),
         bytes, iters, 1, "cross NUMA pair");

  a = (struct copy_arg){buf[1], 1, buf[0], 0, bytes, s0, DEFAULT};
  report(json, "default-0->1", event_gbps(s0, bytes, iters, 1, body_async, &a),
         bytes, iters, 1, "cudaMemcpyAsync(Default)");

  a = (struct copy_arg){host, 0, buf[0], 0, bytes, s0, D2H};
  report(json, "d2h-0", event_gbps(s0, bytes, iters, 1, body_async, &a),
         bytes, iters, 1, "PCIe one direction");

  a = (struct copy_arg){buf[0], 0, host, 0, bytes, s0, H2D};
  report(json, "h2d-0", event_gbps(s0, bytes, iters, 1, body_async, &a),
         bytes, iters, 1, "PCIe one direction");

  /* staged route: GPU0 -> pinned host -> GPU1, synchronised each way */
  {
    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
      check(cudaSetDevice(0), "cudaSetDevice");
      cudaMemcpyAsync(host, buf[0], bytes, D2H, s0);
      check(cudaStreamSynchronize(s0), "cudaStreamSynchronize");
      check(cudaSetDevice(1), "cudaSetDevice");
      cudaMemcpyAsync(buf[1], host, bytes, H2D, s1);
      check(cudaStreamSynchronize(s1), "cudaStreamSynchronize");
    }
    double secs = now_s() - t0;
    double gbps = secs > 0 ? (double)bytes * iters * 2 / secs / 1e9 : -1.0;
    report(json, "staged-0->1", gbps, bytes, iters, 2, "GPU->host->GPU");
  }

  if (json) { fprintf(json, "]\n"); fclose(json); }

  for (int i = 0; i < 4; i++) {
    check(cudaSetDevice(i), "cudaSetDevice");
    cudaFree(buf[i]);
  }
  cudaFreeHost(host);
  printf("done (failures=%d)\n", failures);
  return failures ? 2 : 0;
}
