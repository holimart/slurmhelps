# shellcheck shell=bash
# =============================================================================
#  lib/common.sh — configuration loader for the CLUSTER-SIDE slurmhelps tools
#  (fleet/, image/, bootstrap/, maint/). SOURCED, never executed.
# =============================================================================
#  (The workstation-side bin/* tools read the per-project slurmhelps.json via
#  lib/slurmhelps/config.py; see docs/config.md.)
#
#  Configuration (highest priority first):
#    1. a few historical names from the caller's environment win: AGENTS_DIR,
#       LANGYWRAP_CHECKOUT, AGENT_PROJECT, AGENTS_* footprint, LANE_*;
#    2. an optional override file $SLURMHELPS_PROJECT_ENV (shell assignments;
#       tests and one-off overrides), sourced after the config;
#    3. the project's slurmhelps config (<project>/slurmhelps.json or the
#       "slurmhelps" block of .langywrap/lanes.json), read through
#       `lib/slurmhelps/config.py site-shell`: the "site" section (cluster
#       paths, partitions, MPI env), the "fleet" section (allocation footprint,
#       lane prompt) and everything bin/* uses. The project is
#       $SLURMHELPS_PROJECT, else $AGENT_PROJECT, else found from the cwd.
#  The project's .langywrap/lanes.json carries the fleet blocks
#  (fleet.workset, fleet.node_local_checkout, fleet.session_store,
#  fleet.slurmhelps); the Python helpers read them directly.
# =============================================================================

SLURMHELPS_HOME="${SLURMHELPS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SLURMHELPS_HOME

sh_die() { echo "${SH_PROG:-slurmhelps}: $*" >&2; exit 1; }

# sh_python — the interpreter for config.py/workset.py: the system python3
# (the user toolchain may itself live in an image that is not bound yet).
sh_python() {
  if [ -x /usr/bin/python3 ]; then echo /usr/bin/python3; else command -v python3; fi
}

# sh_load — load the project's slurmhelps config and derive the fleet variables.
sh_load() {
  local proj="${SLURMHELPS_PROJECT:-${AGENT_PROJECT:-}}" out
  [ -n "$proj" ] && { SLURMHELPS_PROJECT="$proj"; export SLURMHELPS_PROJECT; }
  out="$("$(sh_python)" "$SLURMHELPS_HOME/lib/slurmhelps/config.py" site-shell)" \
    || { echo "slurmhelps: cannot read the slurmhelps config of ${proj:-the cwd project}" >&2; return 1; }
  eval "$out"
  if [ -n "${SLURMHELPS_PROJECT_ENV:-}" ]; then
    if [ -f "$SLURMHELPS_PROJECT_ENV" ]; then
      # shellcheck disable=SC1090
      . "$SLURMHELPS_PROJECT_ENV"
    else
      echo "slurmhelps: override file $SLURMHELPS_PROJECT_ENV not found" >&2
      return 1
    fi
    export SLURMHELPS_PROJECT_ENV
  fi
  : "${SH_NODE_TMP:=/tmp}"
  [ -n "${SH_SLURM_BIN:-}" ] && case ":$PATH:" in *":$SH_SLURM_BIN:"*) ;; *) PATH="$PATH:$SH_SLURM_BIN" ;; esac

  # Fleet names used by supervisor.sbatch / ly / lane launchers (historical
  # names; a value already in the environment wins).
  AGENT_PROJECT="${AGENT_PROJECT:-$SH_FLEET_PROJECT}"
  AGENTS_DIR="${AGENTS_DIR:-$SH_AGENTS_DIR}"
  LANGYWRAP_CHECKOUT="${LANGYWRAP_CHECKOUT:-$SH_LANGYWRAP_CHECKOUT}"
  AGENTS_JOBNAME="${AGENTS_JOBNAME:-${SH_FLEET_JOBNAME:-agent-supervisor}}"
  AGENTS_PARTITION="${AGENTS_PARTITION:-$SH_FLEET_PARTITION}"
  AGENTS_TIME="${AGENTS_TIME:-$SH_FLEET_TIME}"
  AGENTS_RENEW_SIGNAL="${AGENTS_RENEW_SIGNAL:-$SH_FLEET_RENEW_SIGNAL}"
  AGENTS_CPUS="${AGENTS_CPUS:-$SH_FLEET_CPUS}"
  AGENTS_MEM="${AGENTS_MEM:-$SH_FLEET_MEM}"
  LANE_MODEL="${LANE_MODEL:-$SH_LANE_MODEL}"
  LANE_PROMPT_FILE="${LANE_PROMPT_FILE:-$SH_LANE_PROMPT_FILE}"
  LANE_EXCLUDE="${LANE_EXCLUDE:-$SH_LANE_EXCLUDE}"
  SH_STORE_SYNC="${SH_STORE_SYNC:-$SLURMHELPS_HOME/fleet/store-sync.py}"
  SH_NODELOCAL_PY="${SH_NODELOCAL_PY:-$SLURMHELPS_HOME/fleet/nodelocal.py}"
  SH_WORKSET="${SH_WORKSET:-$SLURMHELPS_HOME/image/workset.sh}"
  export AGENT_PROJECT AGENTS_DIR LANGYWRAP_CHECKOUT AGENTS_JOBNAME SH_STORE_SYNC SH_NODELOCAL_PY SH_WORKSET
  export SH_WORK_ROOT SH_AGENTS_DIR SH_LANGYWRAP_CHECKOUT SH_ENV_SH SH_USER_BIN SH_IMAGES_DIR \
    SH_FLEET_HOME SH_PI_AGENT_DIR SH_CODEX_HOME SH_XDG_CACHE_HOME SH_NODE_TMP SH_TOOLCHAIN_PROFILE \
    SH_TOOLCHAIN_DIRS SH_NODE_VERSION SH_MPI_ENV SH_ACCOUNT
  return 0
}

# sh_fleet_env — the environment of every fleet process (daemon, lanes, every
# `ly` remote command): toolchain + caches profile, HOME and agent homes on the
# shared filesystem (never the login $HOME), the langywrap venv on PATH. Under
# an active workset (WORKSET_ACTIVE=1, set by `workset.sh exec/enter`) the
# package caches are moved back to node-local tmp after env.sh set them.
sh_fleet_env() {
  if [ -n "${SH_ENV_SH:-}" ] && [ -f "$SH_ENV_SH" ]; then
    # shellcheck disable=SC1090
    . "$SH_ENV_SH"
  fi
  if [ -n "${SH_FLEET_HOME:-}" ]; then
    export HOME="$SH_FLEET_HOME"
    export XDG_CONFIG_HOME="$SH_FLEET_HOME/.config"
    export XDG_DATA_HOME="$SH_FLEET_HOME/.local/share"
    export XDG_STATE_HOME="$SH_FLEET_HOME/.local/state"
  fi
  [ -n "${SH_PI_AGENT_DIR:-}" ] && export PI_CODING_AGENT_DIR="$SH_PI_AGENT_DIR"
  [ -n "${SH_CODEX_HOME:-}" ] && export CODEX_HOME="$SH_CODEX_HOME"
  [ -n "${SH_XDG_CACHE_HOME:-}" ] && export XDG_CACHE_HOME="$SH_XDG_CACHE_HOME"
  local d
  for d in "${HOME:-}" "${PI_CODING_AGENT_DIR:-}" "${CODEX_HOME:-}" "${XDG_CONFIG_HOME:-}" \
           "${XDG_CACHE_HOME:-}" "${XDG_DATA_HOME:-}" "${XDG_STATE_HOME:-}"; do
    [ -n "$d" ] && mkdir -p "$d"
  done
  export PATH="$HOME/.local/bin:${LANGYWRAP_CHECKOUT:+$LANGYWRAP_CHECKOUT/.venv/bin:}${SH_USER_BIN:+$SH_USER_BIN:}$PATH"
  # MPI landmines from the config's site.mpi_env (e.g. OMPI_MCA_pml=ob1), after env.sh.
  # shellcheck disable=SC2086
  [ -n "${SH_MPI_ENV:-}" ] && eval "export $SH_MPI_ENV"
  [ "${WORKSET_ACTIVE:-0}" = 1 ] && sh_workset_cache_env
  return 0
}

# sh_workset_cache_env — package caches on node-local tmp inside a workset
# (the working set is in RAM-backed tmp; a Lustre cache would be the slow path
# again, and a cache inside the tree would be spilled into the image).
sh_workset_cache_env() {
  local c="${WORKSET_CACHE_DIR:-${SH_NODE_TMP:-/tmp}/workset-cache-$(id -u)}"
  export UV_CACHE_DIR="$c/uv" PIP_CACHE_DIR="$c/pip" PYTHONPYCACHEPREFIX="$c/pycache"
  export NPM_CONFIG_CACHE="$c/npm" npm_config_cache="$c/npm" YARN_CACHE_FOLDER="$c/yarn"
  export XDG_CACHE_HOME="$c/xdg"
  mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$NPM_CONFIG_CACHE" "$XDG_CACHE_HOME" 2>/dev/null || true
}

# sh_alloc_value FILE KEY — KEY's value in an alloc.env-style file.
sh_alloc_value() {
  [ -f "$1" ] || return 0
  awk -v k="$2" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$1"
}

# sh_srun_short NAME CMD... — run CMD as a helper job on the short partition
# (builds, tests, image builds; never on the login node).
sh_srun_short() {
  local name="$1"; shift
  srun -p "${SH_PARTITION_SHORT:?config site.partition_short sets SH_PARTITION_SHORT}" ${SH_ACCOUNT:+-A "$SH_ACCOUNT"} \
    -c "$SH_SHORT_CPUS" --mem="$SH_SHORT_MEM" --time="$SH_SHORT_TIME" -J "$name" env -u BASH_ENV "$@"
}
