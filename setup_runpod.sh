#!/usr/bin/env bash
# Phase 4 training pipeline launcher for RunPod (H100 80GB recommended).
#
# PRECONDITIONS on the pod:
#   - Repo cloned (or rsynced) to /workspace/AML and you cd'd into it.
#   - WANDB_API_KEY exported (or pass --wandb-key XXX).
#   - PyTorch + CUDA already in the base image (use RunPod's
#     "PyTorch 2.x" template). If not, install before running.
#
# Usage:
#   export WANDB_API_KEY=...
#   bash setup_runpod.sh
#
# Tunables via env:
#   NUM_WORKERS=12 (default 8) — match pod vCPU count minus 2
#   SKIP_DATASET=1 to skip dataset download (if already present)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export NUM_WORKERS="${NUM_WORKERS:-8}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PIPELINE_LOG="$PROJECT_ROOT/results_4_pipeline.log"
log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }

# ---------- 1. Python deps ----------
log "=== 1/5: install python deps ==="
pip install --quiet --upgrade pip
pip install --quiet \
    kornia \
    albumentations \
    trimesh \
    ultralytics \
    wandb \
    pandas \
    pyyaml \
    opencv-python-headless \
    tqdm \
    gdown

# ---------- 2. Dataset ----------
log "=== 2/5: dataset ==="
if [ -z "${SKIP_DATASET:-}" ] && [ ! -d "datasets/linemod/Linemod_preprocessed" ]; then
    mkdir -p datasets/linemod
    log "downloading LineMod (~1 GB) via gdown..."
    gdown --fuzzy "https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing" \
        -O datasets/linemod/Linemod_preprocessed.zip
    log "extracting..."
    unzip -q -o datasets/linemod/Linemod_preprocessed.zip -d datasets/linemod/
    rm -f datasets/linemod/Linemod_preprocessed.zip
else
    log "dataset present, skipping download."
fi

# ---------- 3. Wandb auth ----------
log "=== 3/5: wandb auth ==="
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null
    log "wandb authenticated."
else
    log "WARNING: WANDB_API_KEY not set — wandb will run in offline mode."
    export WANDB_MODE=offline
fi

# ---------- 4. GPU sanity check ----------
log "=== 4/5: GPU sanity ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only', '| bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)" | tee -a "$PIPELINE_LOG"

# ---------- 5. Pipeline: train + eval for both variants ----------
mkdir -p results_4_main results_4_ext
MAIN_TRAIN_LOG="$PROJECT_ROOT/results_4_main/pipeline_train.log"
MAIN_EVAL_LOG="$PROJECT_ROOT/results_4_main/pipeline_eval.log"
EXT_TRAIN_LOG="$PROJECT_ROOT/results_4_ext/pipeline_train.log"
EXT_EVAL_LOG="$PROJECT_ROOT/results_4_ext/pipeline_eval.log"

run_step() {
    local name="$1" module="$2" logfile="$3"
    log ">>> START $name"
    if python -u -m "$module" 2>&1 | tee -a "$logfile"; then
        log "<<< END   $name (OK)"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        log "<<< END   $name (FAIL rc=$rc)"
        return $rc
    fi
}

log "=== 5/5: pipeline (NUM_WORKERS=$NUM_WORKERS) ==="

if run_step "phase4 MAIN train" "phase4_fusion.main.train" "$MAIN_TRAIN_LOG"; then
    run_step "phase4 MAIN eval"  "phase4_fusion.main.evaluate" "$MAIN_EVAL_LOG" || log "MAIN eval failed (continuing)"
else
    log "MAIN train failed — skipping MAIN eval, going to EXT."
fi

if run_step "phase4 EXT train" "phase4_fusion.extension.train" "$EXT_TRAIN_LOG"; then
    run_step "phase4 EXT eval"  "phase4_fusion.extension.evaluate" "$EXT_EVAL_LOG" || log "EXT eval failed"
else
    log "EXT train failed."
fi

log "=== Pipeline finished. ==="
log "Results in: results_4_main/ and results_4_ext/"
log "Download with:  runpodctl receive <pod-id>:/workspace/AML/results_4_main"
