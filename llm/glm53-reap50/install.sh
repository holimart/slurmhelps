#!/usr/bin/env bash
# Install the GLM-5.3-Flash REAP50 serving stack and checkpoint.
#
# Heavy work (image pull, CUDA build, 100 GB download) is meant to run inside a
# Slurm job, not on a login node. This script is idempotent: existing images,
# checkouts, builds and model shards are reused unless --force is given.
#
# usage: install.sh [--engine sglang|vllm|llamacpp]
#                   [--variant nvfp4|fp8|q4_k_s|q4_k_m|iq4_xs|iq3_m]
#                   [--method auto|container|venv|source]
#                   [--model-root DIR] [--skip-model] [--skip-runtime]
#                   [--cuda-arch 70|89|120] [--dry-run] [--list]
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)

# WORK_ROOT comes from the calling project's config; the helper never hardcodes
# a cluster path. On a node without a project config, $HOME is the fallback.
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
# shellcheck source=config.env
source "$HELPER/config.env"

log()  { printf '[install] %s\n' "$*"; }
warn() { printf '[install] WARN: %s\n' "$*" >&2; }
die()  { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '+ %s\n' "$*"; else "$@"; fi; }

DRY_RUN=0
FORCE=0
SKIP_MODEL=0
SKIP_RUNTIME=0
LIST=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)       ENGINE=$2; shift 2 ;;
    --variant)      VARIANT=$2; shift 2 ;;
    --method)       INSTALL_METHOD=$2; shift 2 ;;
    --model-root)   MODEL_ROOT=$2; shift 2 ;;
    --cuda-arch)    CUDA_ARCH=$2; shift 2 ;;
    --cuda-home)    CUDA_HOME=$2; shift 2 ;;
    --skip-model)   SKIP_MODEL=1; shift ;;
    --skip-runtime) SKIP_RUNTIME=1; shift ;;
    --force)        FORCE=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --list)         LIST=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [ "$LIST" = 1 ]; then
  cat <<'EOF'
variants:
  nvfp4   GLM-5.3-Flash-REAP50-NVFP4-v2    98.2 GiB  Blackwell native, W4A16, quality 0.8386 top-1
  fp8     GLM-5.3-Flash-REAP50-FP8-v2     160.6 GiB  H100/H200 or 4x L40, quality 0.8425 top-1
  q4_k_s  GLM-5.3-Flash-REAP50-Q4_K_S      87.1 GiB  GGUF, llama.cpp, PPL 4.5833
  q4_k_m  GLM-5.3-Flash-REAP50-Q4_K_M      92.5 GiB  GGUF, llama.cpp, PPL 4.5959
  iq4_xs  GLM-5.3-Flash-REAP50-IQ4_XS      82.0 GiB  GGUF, llama.cpp, PPL 4.6002
  iq3_m   GLM-5.3-Flash-REAP50-IQ3_M       67.2 GiB  GGUF, llama.cpp, PPL 4.6253
engines:
  sglang   highest multi-user throughput; v0.5.20+ / lmsysorg/sglang:glm-5.3-flash
  vllm     alternative; SM120 needs a patched image (see README)
  llamacpp single-node / V100 / GGUF fallback; patched fork
EOF
  exit 0
fi

case "$ENGINE" in sglang|vllm|llamacpp) ;; *) die "engine must be sglang, vllm or llamacpp" ;; esac
case "$VARIANT" in nvfp4|fp8|q4_k_s|q4_k_m|iq4_xs|iq3_m) ;; *) die "unknown variant: $VARIANT" ;; esac
case "$VARIANT" in
  nvfp4)     MODEL_SUBDIR=nvfp4;   MODEL_REPO=$HF_REPO_NVFP4; MODEL_FORMAT=safetensors ;;
  fp8)       MODEL_SUBDIR=fp8;     MODEL_REPO=$HF_REPO_FP8;   MODEL_FORMAT=safetensors ;;
  q4_k_s)    MODEL_SUBDIR=gguf; MODEL_REPO=$HF_REPO_GGUF; MODEL_FORMAT=gguf; GGUF_QUANT=Q4_K_S ;;
  q4_k_m)    MODEL_SUBDIR=gguf; MODEL_REPO=$HF_REPO_GGUF; MODEL_FORMAT=gguf; GGUF_QUANT=Q4_K_M ;;
  iq4_xs)    MODEL_SUBDIR=gguf; MODEL_REPO=$HF_REPO_GGUF; MODEL_FORMAT=gguf; GGUF_QUANT=IQ4_XS ;;
  iq3_m)     MODEL_SUBDIR=gguf; MODEL_REPO=$HF_REPO_GGUF; MODEL_FORMAT=gguf; GGUF_QUANT=IQ3_M ;;
esac
MODEL_PATH="$MODEL_ROOT/$MODEL_SUBDIR"
GGUF_FILE="$MODEL_PATH/GLM-5.3-Flash-REAP50-${GGUF_QUANT}.gguf"

log "engine=$ENGINE variant=$VARIANT method=$INSTALL_METHOD"
log "work_root=$WORK_ROOT model_root=$MODEL_ROOT"

mkdir -p "$MODEL_ROOT" "$IMAGE_DIR" "$VENV_DIR" "$SRC_DIR" "$BUILD_DIR" "$RUN_DIR"

UV=$(command -v uv 2>/dev/null || true)
if [ -z "$UV" ] && [ -x "$WORK_ROOT/bin/uv" ]; then UV="$WORK_ROOT/bin/uv"; fi
find_uv() { [ -n "$UV" ] || die "uv not found (expected on PATH or $WORK_ROOT/bin/uv)"; printf '%s\n' "$UV"; }

# ---------------------------------------------------------------------------
# downloader
# ---------------------------------------------------------------------------
find_hf() {
  if command -v hf >/dev/null 2>&1; then command -v hf; return; fi
  if command -v huggingface-cli >/dev/null 2>&1; then command -v huggingface-cli; return; fi
  local dl="$VENV_DIR/downloader"
  if [ ! -x "$dl/bin/hf" ]; then
    find_uv >/dev/null
    run "$UV" venv "$dl" --python 3.12
    run "$UV" pip install --python "$dl/bin/python" "huggingface_hub[hf_transfer]>=0.30"
  fi
  printf '%s\n' "$dl/bin/hf"
}

download_model() {
  if [ "$SKIP_MODEL" = 1 ]; then log "skip model download"; return; fi
  if [ "$MODEL_FORMAT" = safetensors ] && [ -f "$MODEL_PATH/config.json" ] && [ "$FORCE" = 0 ]; then
    log "model already present: $MODEL_PATH"
    return
  fi
  if [ "$MODEL_FORMAT" = gguf ] && [ -f "$GGUF_FILE" ] && [ "$FORCE" = 0 ]; then
    log "GGUF already present: $GGUF_FILE"
    return
  fi
  local hf
  hf=$(find_hf)
  if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; fi
  if [ "$HF_TRANSFER" = 1 ]; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
  mkdir -p "$MODEL_PATH"
  if [ "$MODEL_FORMAT" = gguf ]; then
    log "downloading $MODEL_REPO ($GGUF_QUANT) -> $MODEL_PATH"
    run "$hf" download "$MODEL_REPO" --local-dir "$MODEL_PATH" \
      --include "*${GGUF_QUANT}*" --include "mmproj-*" --include "README.md"
  else
    log "downloading $MODEL_REPO -> $MODEL_PATH"
    # Exclude the unevaluated v3/HOPE subdirectory: it is another ~100 GiB and
    # is not the default mask.
    run "$hf" download "$MODEL_REPO" --local-dir "$MODEL_PATH" \
      --exclude "pass3-hope/*" --exclude "*.md" --exclude "*.png" --exclude "*.jpg"
  fi
  if [ "$MODEL_FORMAT" = gguf ]; then
    [ -f "$GGUF_FILE" ] || warn "expected $GGUF_FILE; check --variant / GGUF_QUANT"
  else
    [ -f "$MODEL_PATH/config.json" ] || warn "config.json missing in $MODEL_PATH"
  fi
}

# ---------------------------------------------------------------------------
# runtimes
# ---------------------------------------------------------------------------
apptainer_bin() {
  if command -v apptainer >/dev/null 2>&1; then command -v apptainer
  elif command -v singularity >/dev/null 2>&1; then command -v singularity
  else return 1; fi
}

pull_image() { # image sif
  local image=$1 sif=$2
  local ap
  ap=$(apptainer_bin) || die "apptainer/singularity not found; use --method venv"
  if [ -f "$sif" ] && [ "$FORCE" = 0 ]; then
    log "image already present: $sif"
    return
  fi
  log "pulling $image -> $sif"
  run "$ap" pull --force "$sif" "docker://$image"
}

install_sglang() {
  case "$INSTALL_METHOD" in
    container)
      pull_image "$SGLANG_IMAGE" "$SGLANG_SIF" ;;
    venv)
      find_uv >/dev/null
      run "$UV" venv "$VENV_DIR/sglang" --python 3.12
      run "$UV" pip install --python "$VENV_DIR/sglang/bin/python" "sglang[all]>=0.5.20,<0.6"
      run "$VENV_DIR/sglang/bin/python" -c 'import sglang; print(sglang.__version__)' ;;
    auto|source)
      if apptainer_bin >/dev/null 2>&1; then
        pull_image "$SGLANG_IMAGE" "$SGLANG_SIF"
      else
        INSTALL_METHOD=venv install_sglang
      fi ;;
  esac
}

install_vllm() {
  case "$INSTALL_METHOD" in
    container)
      pull_image "$VLLM_IMAGE" "$VLLM_SIF" ;;
    venv)
      find_uv >/dev/null
      run "$UV" venv "$VENV_DIR/vllm" --python 3.12
      run "$UV" pip install --python "$VENV_DIR/vllm/bin/python" "vllm>=0.30"
      run "$VENV_DIR/vllm/bin/python" -c 'import vllm; print(vllm.__version__)' ;;
    auto|source)
      if apptainer_bin >/dev/null 2>&1; then
        pull_image "$VLLM_IMAGE" "$VLLM_SIF"
      else
        INSTALL_METHOD=venv install_vllm
      fi ;;
  esac
}

detect_cuda_arch() {
  if [ "$CUDA_ARCH" != native ]; then printf '%s\n' "$CUDA_ARCH"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
    # A login node without a driver prints an error on stdout; ignore it.
    if [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]]; then printf '%s\n' "${cc/./}"; return; fi
  fi
  printf 'native\n'
}

install_llamacpp() {
  [ -d "$LLAMACPP_SRC/.git" ] || run git clone "$LLAMACPP_REPO" "$LLAMACPP_SRC"
  run git -C "$LLAMACPP_SRC" fetch --tags --force origin
  run git -C "$LLAMACPP_SRC" checkout "$LLAMACPP_REF"
  local arch
  arch=$(detect_cuda_arch)
  # CUDA 13 no longer compiles sm_70. If the caller did not choose a toolkit,
  # prefer the newest installed toolkit that still supports the requested arch.
  if [ "$arch" = 70 ] && [ "$CUDA_HOME" = /usr/local/cuda ] && [ -d /usr/local/cuda-12.5 ]; then
    CUDA_HOME=/usr/local/cuda-12.5
  fi
  export CUDA_HOME
  local cmake_args=(-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$arch")
  log "building llama.cpp (CUDA_HOME=$CUDA_HOME arch=$arch)"
  run cmake -S "$LLAMACPP_SRC" -B "$LLAMACPP_BUILD" "${cmake_args[@]}"
  run cmake --build "$LLAMACPP_BUILD" -j "$BUILD_JOBS" --target llama-server
}

write_state() {
  local state="$MODEL_ROOT/install-state.json"
  {
    printf '{\n'
    printf '  "installed_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "engine": "%s",\n' "$ENGINE"
    printf '  "variant": "%s",\n' "$VARIANT"
    printf '  "model_repo": "%s",\n' "$MODEL_REPO"
    printf '  "model_path": "%s",\n' "$MODEL_PATH"
    printf '  "gguf_file": "%s",\n' "$GGUF_FILE"
    printf '  "sglang_sif": "%s",\n' "$SGLANG_SIF"
    printf '  "vllm_sif": "%s",\n' "$VLLM_SIF"
    printf '  "llamacpp_src": "%s",\n' "$LLAMACPP_SRC"
    printf '  "llamacpp_ref": "%s",\n' "$LLAMACPP_REF"
    printf '  "llama_server": "%s"\n' "$LLAMA_SERVER"
    printf '}\n'
  } > "$state"
  log "wrote $state"
}

download_model
if [ "$SKIP_RUNTIME" = 0 ]; then
  case "$ENGINE" in
    sglang)   install_sglang ;;
    vllm)     install_vllm ;;
    llamacpp) install_llamacpp ;;
  esac
fi
if [ "$DRY_RUN" = 0 ]; then write_state; fi

cat <<EOF

Installed.
  engine:  $ENGINE
  variant: $VARIANT
  model:   $MODEL_PATH
  state:   $MODEL_ROOT/install-state.json

Next: serve.sh --engine $ENGINE --variant $VARIANT
      bench.sh --base-url http://127.0.0.1:\${PORT:-30000} --concurrency "1 2 4 8 16 32 64"
EOF
