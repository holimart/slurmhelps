#!/usr/bin/env bash
# setup.sh — Vast.ai instance bring-up for the Strata Titan-RTX benchmark.
# Runs INSIDE the rented container (root). Idempotent-ish; logs to $RESULTS_DIR/setup.log
set -uo pipefail
RESULTS_DIR=${RESULTS_DIR:-/root/results}
MODELS_DIR=${MODELS_DIR:-/root/models}
SETUP_DIR=${SETUP_DIR:-/root}
STRATA_ROOT=${STRATA_ROOT:-/root/strata-engine}
mkdir -p "$RESULTS_DIR" "$MODELS_DIR/dl" "$MODELS_DIR/strata"
export MODELS_DIR RESULTS_DIR
exec > >(tee -a "$RESULTS_DIR/setup.log") 2>&1
log() { printf '[setup] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }


log "host: $(uname -a)"
log "cpu: $(nproc) cores, mem: $(free -g | awk '/Mem:/{print $2}') GiB"
log "disk: $(df -h / | tail -1)"
nvidia-smi || log "WARNING: nvidia-smi failed"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv | tee $RESULTS_DIR/host-nvidia-smi.csv
nvidia-smi -q > $RESULTS_DIR/host-nvidia-smi-q.txt 2>&1 || true
{ echo "date=$(date -u +%FT%TZ)"; uname -a; nproc; free -m; df -h; lscpu | head -25; } > $RESULTS_DIR/host-info.txt 2>&1

# ---- python / hf ------------------------------------------------------------
which python3 pip3 curl git hf || true
python3 -V
python3 -m pip --version || { log "pip missing; trying apt"; apt-get update -qq && apt-get install -y -qq python3-pip; }
python3 -m pip install -q --no-input huggingface_hub hf_transfer || log "WARNING: pip install huggingface_hub failed"
python3 -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
(hf --help >/dev/null 2>&1 && echo "hf CLI present") || log "WARNING: hf CLI missing"

# ---- engine ----------------------------------------------------------------
if [ -x "$STRATA_ROOT/bin/strata" ]; then
  log "engine already installed: $("$STRATA_ROOT/bin/strata" --version 2>&1)"
else
  log "installing engine (cuda13 + cuda12 components)"
  STRATA_ROOT=$STRATA_ROOT LOG=$RESULTS_DIR/strata-install.log \
    bash $SETUP_DIR/install-engine.sh || log "ERROR: install-engine.sh rc=$?"
fi
STRATA_COMPONENTS_DIR=$STRATA_ROOT/components "$STRATA_ROOT/bin/strata" --version 2>&1 || true
STRATA_COMPONENTS_DIR=$STRATA_ROOT/components "$STRATA_ROOT/bin/strata" components engine 2>&1 | tee $RESULTS_DIR/strata-components.txt || true

# ---- models ----------------------------------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
REPO1=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
REPO2=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
DL=$MODELS_DIR/dl

log "downloading IQ3_S pack files (54.8G + 28.8G table)"
hf download "$REPO1" \
  IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --local-dir "$DL" --max-workers 4 || log "ERROR: IQ3_S download rc=$?"

log "downloading Coder IQ1_M shard1 (29.6G)"
hf download "$REPO2" \
  IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --local-dir "$DL" --max-workers 2 || log "ERROR: coder download rc=$?"

# ---- pack dirs (hardlinks; the 28.8G table is one physical file) -----------
mkdir -p $MODELS_DIR/strata/iq3s $MODELS_DIR/strata/coder
ln -f "$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" $MODELS_DIR/strata/iq3s/ 2>/dev/null || true
ln -f "$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" $MODELS_DIR/strata/iq3s/ 2>/dev/null || true
ln -f "$DL/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" $MODELS_DIR/strata/coder/ 2>/dev/null || true
ln -f "$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" $MODELS_DIR/strata/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf 2>/dev/null || true

# ---- verify pinned sizes + sha256 ------------------------------------------
log "verifying pinned sizes + sha256"
python3 - <<'PY' 2>&1 | tee $RESULTS_DIR/verify.txt
import hashlib, os, sys
BASE=os.environ.get("MODELS_DIR", "/root/models") + "/strata"
want = {
 "iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": (54817524224,"4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
 "iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": (28800138432,"316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
 "coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": (29608446496,"e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
 "coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf": (28800138432,"316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
}
bad=0
for rel,(size,sha) in want.items():
    p=os.path.join(BASE,rel)
    if not os.path.exists(p):
        print("MISSING",rel); bad+=1; continue
    st=os.path.getsize(p)
    if st!=size:
        print("SIZE-MISMATCH",rel,st,"want",size); bad+=1; continue
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(1<<24), b""):
            h.update(chunk)
    ok = h.hexdigest()==sha
    print(("VERIFIED " if ok else "SHA-FAIL "),rel,st,h.hexdigest()[:16]+("..." if ok else " want "+sha))
    if not ok: bad+=1
print("bad=%d"%bad)
sys.exit(1 if bad else 0)
PY
log "verify rc=$?"
du -sh $MODELS_DIR/strata/* $MODELS_DIR/dl/* 2>/dev/null
df -h / | tail -1
touch $RESULTS_DIR/SETUP-DONE
log "SETUP-DONE"
