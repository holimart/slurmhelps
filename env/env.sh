# shellcheck shell=bash
# =============================================================================
#  env.sh — user toolchain + cache profile: keep every cache/model/tool store on
#  the shared parallel filesystem under WORK_ROOT, never in $HOME.
# =============================================================================
#  Source of truth: slurmhelps env/env.sh. On a host, $WORK_ROOT/config/env.sh is
#  a 3-line shim that sets WORK_ROOT (and SH_TOOLCHAIN_PROFILE) and sources this
#  file (maint/install-shims.sh writes it), so every project that sources
#  $WORK_ROOT/config/env.sh keeps working.
#
#  Inputs (all optional except WORK_ROOT):
#    WORK_ROOT             per-user root (slurmhelps config site.work_root)
#    SH_NODE_VERSION       node version under $WORK_ROOT/node (default v26.11.1)
#    SH_TOOLCHAIN_PROFILE  site toolchain file (system MPI/MKL, MPI landmines);
#                          default $WORK_ROOT/config/toolchain.sh if present
#  Project- or user-specific extras go in $WORK_ROOT/config/env.local.sh.
# =============================================================================
: "${WORK_ROOT:=${SH_WORK_ROOT:-}}"
if [ -z "$WORK_ROOT" ]; then
  echo "env.sh: WORK_ROOT is not set (source \$WORK_ROOT/config/env.sh, not this file)" >&2
  return 1 2>/dev/null || exit 1
fi
export WORK_ROOT
export TMPDIR="$WORK_ROOT/tmp"

# --- toolchain installed under WORK_ROOT/bin, uv/bin and npm-global/bin ---
for _d in "$WORK_ROOT/bin" "$WORK_ROOT/uv/bin" "$WORK_ROOT/npm-global/bin"; do
  case ":$PATH:" in
    *":$_d:"*) ;;
    *) export PATH="$_d:$PATH" ;;
  esac
done
unset _d

# --- Python packaging / uv / pip ---
export UV_CACHE_DIR="$WORK_ROOT/uv/cache"
export UV_PYTHON_INSTALL_DIR="$WORK_ROOT/uv/python"
export UV_TOOL_DIR="$WORK_ROOT/uv/tools"
export UV_TOOL_BIN_DIR="$WORK_ROOT/uv/bin"
export PIP_CACHE_DIR="$WORK_ROOT/pip/cache"
export PYTHONPYCACHEPREFIX="$WORK_ROOT/cache/pycache"

# --- Hugging Face & model zoos ---
export HF_HOME="$WORK_ROOT/models/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/models/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$WORK_ROOT/datasets/huggingface"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$WORK_ROOT/models/huggingface/sentence-transformers"
export TORCH_HOME="$WORK_ROOT/models/torch"
export TORCH_EXTENSIONS_DIR="$WORK_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$WORK_ROOT/cache/triton"
export KERAS_HOME="$WORK_ROOT/models/keras"
export OLLAMA_MODELS="$WORK_ROOT/models/ollama"
export MPLCONFIGDIR="$WORK_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$WORK_ROOT/cache/numba"
export WANDB_DIR="$WORK_ROOT/cache/wandb"
export WANDB_CACHE_DIR="$WORK_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$WORK_ROOT/config/wandb"
export CUDA_CACHE_PATH="$WORK_ROOT/cache/cuda"

# --- Node / JS tooling (node itself: WORK_ROOT/bin -> WORK_ROOT/node/<ver>) ---
export NODE_HOME="$WORK_ROOT/node/${SH_NODE_VERSION:-v26.11.1}"
export NPM_CONFIG_PREFIX="$WORK_ROOT/npm-global"
export NPM_CONFIG_CACHE="$WORK_ROOT/cache/npm"
export npm_config_cache="$NPM_CONFIG_CACHE"
export YARN_CACHE_FOLDER="$WORK_ROOT/cache/yarn"
export ELECTRON_CACHE="$WORK_ROOT/electron"
export ELECTRON_BUILDER_CACHE="$WORK_ROOT/electron-builder"
export PUPPETEER_CACHE_DIR="$WORK_ROOT/puppeteer"
export PLAYWRIGHT_BROWSERS_PATH="$WORK_ROOT/playwright"

# --- Rust / Go / JVM ---
export CARGO_HOME="$WORK_ROOT/cargo"
export RUSTUP_HOME="$WORK_ROOT/rustup"
export GOPATH="$WORK_ROOT/go"
export GOCACHE="$WORK_ROOT/go/build-cache"
export GOMODCACHE="$WORK_ROOT/go/pkg/mod"
export GRADLE_USER_HOME="$WORK_ROOT/gradle"

# --- Conda / Poetry / pre-commit / Julia ---
export CONDA_PKGS_DIRS="$WORK_ROOT/conda-pkgs"
export POETRY_CACHE_DIR="$WORK_ROOT/poetry"
export POETRY_VIRTUALENVS_PATH="$WORK_ROOT/poetry/venvs"
export PRE_COMMIT_HOME="$WORK_ROOT/precommit"
export JULIA_DEPOT_PATH="$WORK_ROOT/julia"

# --- Agents ---
export CLAUDE_CONFIG_DIR="$WORK_ROOT/home/claude"

# --- Misc tool databases ---
export DOCKER_CONFIG="$WORK_ROOT/config/docker"
export NUGET_PACKAGES="$WORK_ROOT/cache/nuget"

# --- site toolchain (system MPI/BLAS, MPI landmines): config site.toolchain_profile ---
_tc="${SH_TOOLCHAIN_PROFILE:-$WORK_ROOT/config/toolchain.sh}"
[ -f "$_tc" ] && . "$_tc"
unset _tc

# --- user/project extras (not part of slurmhelps) ---
[ -f "$WORK_ROOT/config/env.local.sh" ] && . "$WORK_ROOT/config/env.local.sh"

