# 2026-10-22 — Vulkan smoke on 2× L40S 48 GB

Short smoke to see whether the GLM5-Next REAP50 GGUF can run on the Vulkan
backend before spending a 4×L40 allocation.

## Build

- Vulkan SDK 1.4.363 (LunarG) extracted to
  `/work/ffa/holecem/tools/vulkansdk/x86_64`; provides `glslc`.
- `cmake -DGGML_CUDA=OFF -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`.
- `llama-server` built at
  `/work/ffa/holecem/build/llama-vulkan/bin/llama-server`.

The fork implements `ggml_mhc_sinkhorn` in CPU and CUDA only. There is no
Vulkan kernel for it, so the scheduler must fall back to the CPU for that
operator. This smoke was expected to test whether that works and how slow it
is.

## Run

`uk-g6-05`, 2× L40S 48 GB (96 GB), 32 CPUs, 256 GiB, 30 min.
IQ4_XS 82 GiB, `-ngl 99 --split-mode layer -np 1 -c 8192 -b 4096 -ub 1024
-fa off -ctk f16 -ctv f16 --no-mmap`.

## IQ4_XS: failed on memory, before token generation

The model loaded across both cards:

```text
load_tensors:      Vulkan0 model buffer size = 40826.97 MiB
load_tensors:      Vulkan1 model buffer size = 42812.37 MiB
load_tensors:  Vulkan_Host model buffer size =   321.41 MiB
```

Then the first graph compute failed:

```text
terminate called after throwing an instance of 'vk::SystemError'
  what():  vk::Device::allocateMemory: ErrorOutOfDeviceMemory
```

Backtrace: `ggml_vk_preallocate_buffers` → `ggml_vk_mul_mat_q_f16` →
`ggml_vk_build_graph` → `ggml_backend_vk_graph_compute` → `llama_decode`.

This is **Vulkan compute-buffer OOM**, not a missing mHC operator. With ~5 GB
left per 48 GB card, Vulkan's preallocated matmul buffers do not fit. A 4× L40
node (192 GB) has enough headroom, or 2× L40 with IQ3_M (67 GiB) would free
~15 GB.

## IQ3_M: Vulkan works, but slowly

The same server with **IQ3_M (67.2 GiB)** fits 2× L40S and completes a 4K
prompt + 8-token generation:

```text
Vulkan0 model buffer = 33583.59 MiB
Vulkan1 model buffer = 34938.18 MiB
Vulkan0 compute      =  2660 MiB
Vulkan1 compute      =  3146 MiB
free after load      = 6003 / 5665 MiB
```

Measured by the smoke script:

| Metric | Value |
|---|---:|
| Prompt | 4K |
| TTFT | **20.32 s** |
| Output | 8 tokens |
| Output rate | **0.4 tok/s** |
| Total (prompt+output) | 201.8 tok/s |
| TPOT | 122.9 ms |

This proves the Vulkan backend can execute GLM5-Next with IQ3_M, including the
CPU fallback for `ggml_mhc_sinkhorn`. It is not a performance path: 4K TTFT is
20 s and decode is ~8 tok/s before batching. The CPU mHC fallback is visible
in the huge gap between the 4K prefill and the token rate.

No 1M or 4× L40 Vulkan run was possible in this test; 96 GB is already tight
at 4K with IQ4_XS and works at 4K with IQ3_M.

## Status of the 4× L40 run

`fuuk-13/14/15` (4× L40, 192 GB) were fully allocated at the time, and
`ampere01` had all three L40s allocated. The Vulkan smoke could not be run on
four cards. It remains to do when a 4× L40 node is free:

```bash
LLAMA_SERVER=/work/ffa/holecem/build/llama-vulkan/bin/llama-server \
LLAMA_FA=off LLAMA_KV_TYPE=f16 CTX_TOTAL=8192 MAX_RUNNING=1 \
sbatch -p gpu-ffa -w fuuk-13 --gres=gpu:4 --cpus-per-task=32 --mem=256G \
  --time=00:30:00 --wrap='bash .../vulkan-smoke.sh'
```

The same binary can be used for the CUDA 4×L40 baseline with the sm_89
`llama-server`; that is the more representative measurement.
