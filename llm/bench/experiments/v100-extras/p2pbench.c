/* p2pbench.c — measure peer (NVLink) vs PCIe host-staged copy bandwidth.
 *
 * Built with plain gcc; no CUDA toolkit headers or nvcc are needed because
 * the CUDA *driver* API is loaded with dlopen() from libcuda.so.1.  This is
 * deliberate: the node's only nvcc is CUDA 13.4, which no longer emits sm_70
 * code, and the vLLM venv's torch is not a trustworthy measurement device.
 *
 * Tests (payload 256 MiB unless --mib says otherwise):
 *   local-d2d        GPU0 -> GPU0        (HBM sanity)
 *   peer-0->1        cuMemcpyPeerAsync   (same NUMA pair; NVLink if enabled)
 *   peer-0->2        cuMemcpyPeerAsync   (cross NUMA pair)
 *   d2h-0 / h2d-0    GPU0 <-> pinned host (PCIe, one direction)
 *   staged-0->1      GPU0 -> host -> GPU1 (what a no-P2P fallback pays)
 *
 * Peer bandwidth is timed with CUDA events around N async copies; the host
 * round trip is timed with CLOCK_MONOTONIC around N synchronised pairs.
 * Effective GB/s for the staged route counts both directions.
 *
 * usage: p2pbench [--mib 256] [--iters 20] [--json out.json]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_STREAM_DEFAULT 0

static CUresult (*cuInit)(unsigned);
static CUresult (*cuDeviceGet)(CUdevice *, int);
static CUresult (*cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*cuCtxSetCurrent)(CUcontext);
static CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned);
static CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice);
static CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*cuMemFree)(CUdeviceptr);
static CUresult (*cuMemHostAlloc)(void **, size_t, unsigned);
static CUresult (*cuMemFreeHost)(void *);
static CUresult (*cuMemcpyPeerAsync)(CUdeviceptr, CUcontext, CUdeviceptr,
                                     CUcontext, size_t, CUstream);
static CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
static CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*cuStreamCreate)(CUstream *, unsigned);
static CUresult (*cuStreamSynchronize)(CUstream);
static CUresult (*cuEventCreate)(CUevent *, unsigned);
static CUresult (*cuEventRecord)(CUevent, CUstream);
static CUresult (*cuEventSynchronize)(CUevent);
static CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent);
static CUresult (*cuCtxSynchronize)(void);

static void *lib;
static int failures;
static int json_count;

static void *sym(const char *name) {
  void *p = dlsym(lib, name);
  if (!p) {
    fprintf(stderr, "missing driver symbol %s\n", name);
    exit(2);
  }
  return p;
}

static void check(CUresult rc, const char *what) {
  if (rc != CUDA_SUCCESS) {
    fprintf(stderr, "%s failed: CUresult=%d\n", what, rc);
    failures++;
  }
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* --- event-timed copy bodies (one copy per call) -------------------------- */
struct peer_arg { CUdeviceptr dst; CUcontext dctx; CUdeviceptr src;
                  CUcontext sctx; size_t bytes; CUstream s; };

static void body_peer(void *p) {
  struct peer_arg *a = p;
  cuMemcpyPeerAsync(a->dst, a->dctx, a->src, a->sctx, a->bytes, a->s);
}
static void body_d2d(void *p) {
  struct peer_arg *a = p;
  cuMemcpyDtoDAsync(a->dst, a->src, a->bytes, a->s);
}
static void body_d2h(void *p) {
  struct peer_arg *a = p;
  cuMemcpyDtoHAsync((void *)a->dst, a->src, a->bytes, a->s);
}
static void body_h2d(void *p) {
  struct peer_arg *a = p;
  cuMemcpyHtoDAsync(a->dst, (const void *)a->src, a->bytes, a->s);
}

static double event_gbps(CUstream s, size_t bytes, int iters, int multiply,
                         void (*body)(void *), void *arg) {
  CUevent a, b;
  float ms = 0.0f;
  check(cuEventCreate(&a, 0), "cuEventCreate");
  check(cuEventCreate(&b, 0), "cuEventCreate");
  check(cuEventRecord(a, s), "cuEventRecord");
  for (int i = 0; i < iters; i++) body(arg);
  check(cuEventRecord(b, s), "cuEventRecord");
  check(cuEventSynchronize(b), "cuEventSynchronize");
  check(cuEventElapsedTime(&ms, a, b), "cuEventElapsedTime");
  if (ms <= 0.0f) return -1.0;
  return (double)bytes * iters * multiply / (ms / 1000.0) / 1e9;
}

static void report(FILE *json, const char *test, double gbps, size_t bytes,
                   int iters, int multiply, const char *note) {
  printf("%-12s %8.2f GB/s  (%zu MiB x %d%s)%s%s\n", test, gbps,
         bytes >> 20, iters,
         multiply == 2 ? ", both directions" : "",
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
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mib") && i + 1 < argc) mib = strtoul(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else { fprintf(stderr, "usage: %s [--mib N] [--iters N] [--json FILE]\n",
                   argv[0]); return 2; }
  }
  const size_t bytes = mib << 20;

  lib = dlopen("libcuda.so.1", RTLD_NOW);
  if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
  if (!lib) { fprintf(stderr, "cannot load libcuda.so.1: %s\n", dlerror()); return 2; }
  cuInit = sym("cuInit"); cuDeviceGet = sym("cuDeviceGet");
  cuDeviceGetName = sym("cuDeviceGetName");
  cuDevicePrimaryCtxRetain = sym("cuDevicePrimaryCtxRetain");
  cuCtxSetCurrent = sym("cuCtxSetCurrent");
  cuCtxEnablePeerAccess = sym("cuCtxEnablePeerAccess");
  cuDeviceCanAccessPeer = sym("cuDeviceCanAccessPeer");
  cuMemAlloc = sym("cuMemAlloc"); cuMemFree = sym("cuMemFree");
  cuMemHostAlloc = sym("cuMemHostAlloc"); cuMemFreeHost = sym("cuMemFreeHost");
  cuMemcpyPeerAsync = sym("cuMemcpyPeerAsync");
  cuMemcpyDtoDAsync = sym("cuMemcpyDtoDAsync");
  cuMemcpyDtoHAsync = sym("cuMemcpyDtoHAsync");
  cuMemcpyHtoDAsync = sym("cuMemcpyHtoDAsync");
  cuStreamCreate = sym("cuStreamCreate");
  cuStreamSynchronize = sym("cuStreamSynchronize");
  cuEventCreate = sym("cuEventCreate"); cuEventRecord = sym("cuEventRecord");
  cuEventSynchronize = sym("cuEventSynchronize");
  cuEventElapsedTime = sym("cuEventElapsedTime");
  cuCtxSynchronize = sym("cuCtxSynchronize");

  CUdevice dev[4];
  CUcontext ctx[4];
  CUdeviceptr buf[4];
  char name[128];
  check(cuInit(0), "cuInit");
  for (int i = 0; i < 4; i++) {
    check(cuDeviceGet(&dev[i], i), "cuDeviceGet");
    check(cuDevicePrimaryCtxRetain(&ctx[i], dev[i]), "cuDevicePrimaryCtxRetain");
    cuDeviceGetName(name, sizeof(name), dev[i]);
    printf("GPU %d: %s\n", i, name);
    check(cuCtxSetCurrent(ctx[i]), "cuCtxSetCurrent");
    check(cuMemAlloc(&buf[i], bytes), "cuMemAlloc");
  }
  int can01 = -1, can02 = -1;
  cuDeviceCanAccessPeer(&can01, dev[0], dev[1]);
  cuDeviceCanAccessPeer(&can02, dev[0], dev[2]);
  printf("canAccessPeer 0->1=%d 0->2=%d\n", can01, can02);

  CUresult e01, e02, e10;
  cuCtxSetCurrent(ctx[0]);
  e01 = cuCtxEnablePeerAccess(ctx[1], 0);
  e02 = cuCtxEnablePeerAccess(ctx[2], 0);
  cuCtxSetCurrent(ctx[1]);
  e10 = cuCtxEnablePeerAccess(ctx[0], 0);
  printf("enablePeerAccess 0->1=%d 0->2=%d 1->0=%d (0 = success, "
         "704 = already enabled)\n", e01, e02, e10);

  void *host = NULL;
  check(cuMemHostAlloc(&host, bytes, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
  memset(host, 0x5a, bytes);  /* fault the pinned pages in before timing */

  cuCtxSetCurrent(ctx[0]);
  CUstream s0, s1;
  check(cuStreamCreate(&s0, CU_STREAM_DEFAULT), "cuStreamCreate");
  cuCtxSetCurrent(ctx[1]);
  check(cuStreamCreate(&s1, CU_STREAM_DEFAULT), "cuStreamCreate");

  FILE *json = json_path ? fopen(json_path, "w") : NULL;
  if (json) fprintf(json, "[");

  struct peer_arg a;

  /* warm-up (first touch of both allocations) */
  cuCtxSetCurrent(ctx[0]);
  a = (struct peer_arg){buf[1], ctx[1], buf[0], ctx[0], bytes, s0};
  body_peer(&a);
  cuStreamSynchronize(s0);

  a = (struct peer_arg){buf[0], ctx[0], buf[0], ctx[0], bytes, s0};
  report(json, "local-d2d", event_gbps(s0, bytes, iters, 1, body_d2d, &a),
         bytes, iters, 1, "HBM sanity");

  a = (struct peer_arg){buf[1], ctx[1], buf[0], ctx[0], bytes, s0};
  report(json, "peer-0->1", event_gbps(s0, bytes, iters, 1, body_peer, &a),
         bytes, iters, 1, "same NUMA pair");

  a = (struct peer_arg){buf[2], ctx[2], buf[0], ctx[0], bytes, s0};
  report(json, "peer-0->2", event_gbps(s0, bytes, iters, 1, body_peer, &a),
         bytes, iters, 1, "cross NUMA pair");

  a = (struct peer_arg){(CUdeviceptr)host, ctx[0], buf[0], ctx[0], bytes, s0};
  report(json, "d2h-0", event_gbps(s0, bytes, iters, 1, body_d2h, &a),
         bytes, iters, 1, "PCIe one direction");

  cuCtxSetCurrent(ctx[0]);
  a = (struct peer_arg){buf[0], ctx[0], (CUdeviceptr)host, ctx[0], bytes, s0};
  report(json, "h2d-0", event_gbps(s0, bytes, iters, 1, body_h2d, &a),
         bytes, iters, 1, "PCIe one direction");

  /* staged route: GPU0 -> pinned host -> GPU1, synchronised each way */
  {
    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
      cuCtxSetCurrent(ctx[0]);
      cuMemcpyDtoHAsync(host, buf[0], bytes, s0);
      cuStreamSynchronize(s0);
      cuCtxSetCurrent(ctx[1]);
      cuMemcpyHtoDAsync(buf[1], host, bytes, s1);
      cuStreamSynchronize(s1);
    }
    double secs = now_s() - t0;
    double gbps = secs > 0 ? (double)bytes * iters * 2 / secs / 1e9 : -1.0;
    report(json, "staged-0->1", gbps, bytes, iters, 2, "GPU->host->GPU");
  }

  if (json) { fprintf(json, "]\n"); fclose(json); }

  cuStreamSynchronize(s0);
  for (int i = 0; i < 4; i++) {
    cuCtxSetCurrent(ctx[i]);
    cuMemFree(buf[i]);
  }
  cuMemFreeHost(host);
  printf("done (failures=%d)\n", failures);
  return failures ? 1 : 0;
}
