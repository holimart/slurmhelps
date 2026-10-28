// hbm-bw.cpp — minimal HIP HBM bandwidth baseline for the MI210 campaign.
//
// Build: hipcc -O3 --offload-arch=gfx90a hbm-bw.cpp -o hbm-bw
// Run:   HIP_VISIBLE_DEVICES=0 ./hbm-bw [n_floats] [reps]
//
// Vectorised float4 kernels, grid-stride over 4096 blocks.  Reports:
//   copy   d2d copy         (reads N, writes N -> 2N bytes)
//   triad  a = b + s*c      (reads 2N, writes N -> 3N bytes)
//   read   sum over a       (reads N; the decode-like weight-stream pattern)
// Bytes are logical traffic; effective GB/s = bytes/time.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void copy_kernel(float4 * __restrict__ a, const float4 * __restrict__ b, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) a[i] = b[i];
}
__global__ void triad_kernel(float4 * __restrict__ a, const float4 * __restrict__ b, const float4 * __restrict__ c, float s, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 x = b[i], y = c[i];
        a[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
    }
}
__global__ void read_kernel(const float4 * __restrict__ a, float * __restrict__ out, size_t n4) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 x = a[i];
        acc += x.x + x.y + x.z + x.w;
    }
    if (acc == 12345.678f) out[0] = acc; // never true; keeps the read live
}

int main(int argc, char ** argv) {
    size_t n = (argc > 1) ? strtoull(argv[1], nullptr, 0) : (size_t)1 << 28; // 1 GiB of floats
    n &= ~(size_t)3;
    int reps = (argc > 2) ? atoi(argv[2]) : 10;
    const size_t n4 = n / 4;
    const size_t bytes = n * sizeof(float);
    float *a, *b, *c, *out;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMalloc(&c, bytes));
    HIP_CHECK(hipMalloc(&out, sizeof(float)));
    HIP_CHECK(hipMemset(a, 0, bytes));
    HIP_CHECK(hipMemset(b, 1, bytes));
    HIP_CHECK(hipMemset(c, 2, bytes));

    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, dev));
    printf("device=%d name=%s gcnArchName=%s mem=%.1f GiB blocks=%d\n", dev, p.name, p.gcnArchName,
           p.totalGlobalMem / 1073741824.0, 4096);

    const int block = 256, grid = 4096;
    struct { const char *name; double mult; } modes[] = {{"copy", 2.0}, {"triad", 3.0}, {"read", 1.0}};
    for (auto m : modes) {
        if (m.name[0] == 'c') copy_kernel<<<grid, block>>>((float4*)a, (const float4*)b, n4);
        else if (m.name[0] == 't') triad_kernel<<<grid, block>>>((float4*)a, (const float4*)b, (const float4*)c, 1.5f, n4);
        else read_kernel<<<grid, block>>>((const float4*)b, out, n4);
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
        for (int r = 0; r < reps; ++r) {
            if (m.name[0] == 'c') copy_kernel<<<grid, block>>>((float4*)a, (const float4*)b, n4);
            else if (m.name[0] == 't') triad_kernel<<<grid, block>>>((float4*)a, (const float4*)b, (const float4*)c, 1.5f, n4);
            else read_kernel<<<grid, block>>>((const float4*)b, out, n4);
        }
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        double gbs = (m.mult * (double)bytes * reps) / (ms * 1e6);
        printf("%-6s n=%zu reps=%d time_ms=%.2f GB/s=%.1f\n", m.name, n, reps, ms, gbs);
    }
    HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(c)); HIP_CHECK(hipFree(out));
    return 0;
}
