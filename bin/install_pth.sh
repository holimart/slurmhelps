#!/usr/bin/env bash
# Install slurmhelps.pth (naming this checkout's lib/) into a Python venv, so `import hpc_partial`
# and `import slurmhelps` work for every interpreter of that venv, even when PYTHONPATH is set inline.
#   ../slurmhelps/bin/install_pth.sh [VENV]      # default: .venv of the current directory
set -euo pipefail
LIB=$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)
VENV=${1:-.venv}
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "install_pth.sh: no interpreter at $PY" >&2; exit 2; }
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$LIB" > "$SP/slurmhelps.pth"
"$PY" -c 'import hpc_partial, slurmhelps.config' && echo "installed $SP/slurmhelps.pth -> $LIB"
