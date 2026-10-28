#!/usr/bin/env bash
# setup3090.sh — Vast.ai RTX 3090 campaign bring-up (1x / 2x / 4x).
# Downloads only the packs in $PACKS_TO_FETCH (iq3s coder q2_0), pinned revisions,
# one shared shard-2 (the 28.8 GB PLE table) hardlinked into each pack dir.
set -uo pipefail
RESULTS_DIR=${RESULTS_DIR:-/root/results}
MODELS_DIR=${MODELS_DIR:-/root/models}
SETUP_DIR=${SETUP_DIR:-/root}
STRATA_ROOT=${STRATA_ROOT:-/root/strata-engine}
mkdir -p "$RESULTS_DIR" "$MODELS_DIR/dl" "$MODELS_DIR/strata"
export MODELS_DIR RESULTS_DIR
exec > >(tee -a "$RESULTS_DIR/setup.log") 2>&1
log() { printf '[setup] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

FETCH=${PACKS_TO_FETCH:-"iq3s coder"}
DL=$MODELS_DIR/dl

log "packs to fetch: $FETCH"
log "host: $(uname -a)"
log "cpu: $(nproc) cores, mem: $(free -g | awk '/Mem:/{print $2}') GiB"
log "disk: $(df -h / | tail -1)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv | tee $RESULTS_DIR/host-nvidia-smi.csv
nvidia-smi -q > $RESULTS_DIR/host-nvidia-smi-q.txt 2>&1 || true
{ echo "date=$(date -u +%FT%TZ)"; uname -a; nproc; free -m; df -h; lscpu | head -25; } > $RESULTS_DIR/host-info.txt 2>&1

python3 -V
python3 -m pip install -q --no-input huggingface_hub || log "WARNING: pip install failed"
python3 -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

if [ -x "$STRATA_ROOT/bin/strata" ]; then
  log "engine already installed"
else
  log "installing engine (cuda13 + cuda12)"
  STRATA_ROOT=$STRATA_ROOT LOG=$RESULTS_DIR/strata-install.log bash $SETUP_DIR/install-engine.sh || log "ERROR: install rc=$?"
fi
STRATA_COMPONENTS_DIR=$STRATA_ROOT/components "$STRATA_ROOT/bin/strata" --version 2>&1 || true
STRATA_COMPONENTS_DIR=$STRATA_ROOT/components "$STRATA_ROOT/bin/strata" components engine 2>&1 | tee $RESULTS_DIR/strata-components.txt || true

export HF_HUB_DISABLE_TELEMETRY=1
REV1=ed59f92082b1e93c0e96d60a8b11aab089b52f09   # IQ3_S / Q2_0 repo, 2026-09-29
REV2=5348543e0147355ac9cbcb031184a3546350988e   # Coder repo, 2026-09-29
REPO1=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
REPO2=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF

need_table=0
for p in $FETCH; do
  case "$p" in
    iq3s)
      log "downloading IQ3_S shard1 + table (pinned ${REV1:0:8})"
      hf download "$REPO1" \
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --revision "$REV1" --local-dir "$DL" --max-workers 4 || log "ERROR: iq3s download rc=$?"
      need_table=1 ;;
    coder)
      log "downloading Coder IQ1_M shard1 (pinned ${REV2:0:8})"
      hf download "$REPO2" \
        IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
        --revision "$REV2" --local-dir "$DL" --max-workers 2 || log "ERROR: coder download rc=$?"
      need_table=1 ;;
    q2_0)
      log "downloading Q2_0 shard1 + table (pinned ${REV1:0:8})"
      hf download "$REPO1" \
        Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
        Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
        --revision "$REV1" --local-dir "$DL" --max-workers 4 || log "ERROR: q2_0 download rc=$?"
      need_table=1 ;;
    *) log "ERROR: unknown pack $p" ;;
  esac
done

# pack dirs (hardlinks; one physical 28.8 GB table for all packs)
mkdir -p $MODELS_DIR/strata/iq3s $MODELS_DIR/strata/coder $MODELS_DIR/strata/q2_0
TABLE=""
[ -f "$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" ] && TABLE="$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
[ -z "$TABLE" ] && [ -f "$DL/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" ] && TABLE="$DL/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"

for p in $FETCH; do
  case "$p" in
    iq3s)
      ln -f "$DL/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" $MODELS_DIR/strata/iq3s/ 2>/dev/null || true
      ln -f "$TABLE" $MODELS_DIR/strata/iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf 2>/dev/null || true ;;
    coder)
      ln -f "$DL/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" $MODELS_DIR/strata/coder/ 2>/dev/null || true
      ln -f "$TABLE" $MODELS_DIR/strata/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf 2>/dev/null || true ;;
    q2_0)
      ln -f "$DL/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" $MODELS_DIR/strata/q2_0/ 2>/dev/null || true
      ln -f "$TABLE" $MODELS_DIR/strata/q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf 2>/dev/null || true ;;
  esac
done

log "verifying pinned sizes + sha256 ($FETCH)"
PACKS_TO_FETCH="$FETCH" python3 - <<'PY' 2>&1 | tee $RESULTS_DIR/verify.txt
import hashlib, os, sys
BASE = os.environ.get("MODELS_DIR", "/root/models") + "/strata"
TABLE = (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")
SPEC = {
 "iq3s": [("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", 54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
          ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",) + TABLE],
 "coder": [("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", 29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
           ("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",) + TABLE],
 "q2_0": [("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", 37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
          ("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",) + TABLE],
}
want = os.environ.get("PACKS_TO_FETCH", "iq3s coder").split()
bad = 0
for pack in want:
    for f, size, sha in SPEC[pack]:
        p = os.path.join(BASE, pack, f)
        if not os.path.exists(p):
            print("MISSING", pack, f); bad += 1; continue
        st = os.path.getsize(p)
        if st != size:
            print("SIZE-MISMATCH", pack, f, st, "want", size); bad += 1; continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        ok = h.hexdigest() == sha
        print(("VERIFIED " if ok else "SHA-FAIL "), pack + "/" + f, st, h.hexdigest()[:16] + ("..." if ok else " want " + sha))
        if not ok: bad += 1
print("bad=%d" % bad)
sys.exit(1 if bad else 0)
PY
log "verify rc=$?"
du -sh $MODELS_DIR/strata/* 2>/dev/null
df -h / | tail -1
touch $RESULTS_DIR/SETUP-DONE
log "SETUP-DONE"
