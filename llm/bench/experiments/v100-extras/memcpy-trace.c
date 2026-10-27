/* memcpy-trace.c — LD_PRELOAD shim that records CUDA runtime copy calls.
 *
 * Purpose (R2): decide *which transport* the layer-split stage hand-off uses.
 * The engine links libcudart.so.12, so interposing the runtime entry points is
 * enough to see whether a 4-GPU decode/prefill moves activations with
 * cudaMemcpyPeerAsync (NVLink, when peer access is on) or with
 * cudaMemcpyAsync(H2D/D2H/Default) (PCIe or a host-staged fallback), and how
 * many bytes per call.
 *
 * Build:  gcc -O2 -shared -fPIC -o memcpy-trace.so memcpy-trace.c -ldl -lpthread
 * Use:    STRATA_MEMCPY_TRACE=/tmp/trace.tsv LD_PRELOAD=./memcpy-trace.so strata ...
 *
 * Output: one TSV line per call
 *   realtime_ns  fn  kind  bytes  src_hex  dst_hex  rc
 * plus `enable_peer` / `can_access_peer` rows.  No file is written when
 * STRATA_MEMCPY_TRACE is unset (the shim is then a pure pass-through).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void *cudaStream_t;
typedef int cudaError_t;

enum { H2H = 0, H2D = 1, D2H = 2, D2D = 3, DEFAULT = 4 };

typedef cudaError_t (*fn3)(void *, const void *, size_t);
typedef cudaError_t (*fn4)(void *, const void *, size_t, int);
typedef cudaError_t (*fn5)(void *, const void *, size_t, int, cudaStream_t);
typedef cudaError_t (*peer_fn)(void *, int, const void *, int, size_t,
                               cudaStream_t);
typedef cudaError_t (*peer_sync_fn)(void *, int, const void *, int, size_t);
typedef cudaError_t (*enable_fn)(int, unsigned);
typedef cudaError_t (*can_fn)(int *, int, int);

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *out;
static int initialised;
static int disabled;

static long long now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void init(void) {
  const char *path = getenv("STRATA_MEMCPY_TRACE");
  if (!path || !*path) { disabled = 1; return; }
  out = fopen(path, "a");
  if (!out) { disabled = 1; return; }
  setvbuf(out, NULL, _IOLBF, 0);
  initialised = 1;
}

static void log_row(const char *fn, long long kind, size_t bytes,
                    const void *src, const void *dst, long long rc) {
  if (disabled) return;
  if (!initialised) init();
  if (disabled || !out) return;
  pthread_mutex_lock(&lock);
  fprintf(out, "%lld\t%s\t%lld\t%zu\t%p\t%p\t%lld\n", now_ns(), fn, kind,
          bytes, src, dst, rc);
  pthread_mutex_unlock(&lock);
}

static void *next(const char *name) {
  void *p = dlsym(RTLD_NEXT, name);
  if (!p) {
    static int warned;
    if (!warned) { fprintf(stderr, "[memcpy-trace] missing %s\n", name); warned = 1; }
  }
  return p;
}

static const char *kind_name(int kind) {
  switch (kind) {
    case H2H: return "h2h";
    case H2D: return "h2d";
    case D2H: return "d2h";
    case D2D: return "d2d";
    case DEFAULT: return "default";
    default: return "?";
  }
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, int kind) {
  static fn4 real;
  if (!real) real = (fn4)next("cudaMemcpy");
  cudaError_t rc = real(dst, src, n, kind);
  log_row(kind_name(kind), kind, n, src, dst, rc);
  return rc;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, int kind,
                            cudaStream_t stream) {
  static fn5 real;
  if (!real) real = (fn5)next("cudaMemcpyAsync");
  cudaError_t rc = real(dst, src, n, kind, stream);
  log_row(kind_name(kind), kind, n, src, dst, rc);
  return rc;
}

cudaError_t cudaMemcpyPeer(void *dst, int dst_dev, const void *src,
                           int src_dev, size_t n) {
  static peer_sync_fn real;
  if (!real) real = (peer_sync_fn)next("cudaMemcpyPeer");
  cudaError_t rc = real(dst, dst_dev, src, src_dev, n);
  log_row("cudaMemcpyPeer", -1, n, src, dst, rc);
  return rc;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dst_dev, const void *src,
                                int src_dev, size_t n, cudaStream_t stream) {
  static peer_fn real;
  if (!real) real = (peer_fn)next("cudaMemcpyPeerAsync");
  cudaError_t rc = real(dst, dst_dev, src, src_dev, n, stream);
  log_row("cudaMemcpyPeerAsync", -1, n, src, dst, rc);
  return rc;
}

cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned flags) {
  static enable_fn real;
  if (!real) real = (enable_fn)next("cudaDeviceEnablePeerAccess");
  cudaError_t rc = real(peer, flags);
  log_row("enable_peer", peer, 0, NULL, NULL, rc);
  return rc;
}

cudaError_t cudaDeviceCanAccessPeer(int *can, int dev, int peer) {
  static can_fn real;
  if (!real) real = (can_fn)next("cudaDeviceCanAccessPeer");
  cudaError_t rc = real(can, dev, peer);
  log_row("can_access_peer", peer, 0, NULL, NULL,
          rc == 0 ? (can ? *can : -1) : rc);
  return rc;
}
