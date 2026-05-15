#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/benchmark_work}"
WILSON="$ROOT/wilson"
CHAIN_ID="${CHAIN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOB_ROOT="$WILSON/fast_58_benchmark_chain_${CHAIN_ID}"
OUT_ROOT="$JOB_ROOT/preprocess"
RUN_CWD="$JOB_ROOT/fast_zipparams"
ZIP_SRC="$JOB_ROOT/source_zip"
ZIP_REPRO="$ZIP_SRC/reproduce_FAST"
LOG_DIR="$WILSON/logs"
DATASET_NAME="${DATASET_NAME:-fast_benchmark_58}"
NPZ="$OUT_ROOT/${DATASET_NAME}.npz"
WORKERS="${WORKERS:-3}"
FORCE_ARGS=()
if [[ "${FORCE_PREPROCESS:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

mkdir -p "$JOB_ROOT" "$OUT_ROOT" "$RUN_CWD" "$LOG_DIR" "$ZIP_SRC"
unzip -q -o /media/datasets/DSO_PP2025/reproduce_FAST.zip \
  'reproduce_FAST/*.py' \
  -d "$ZIP_SRC"

echo "START $(date '+%F %T %z')"
echo "JOB_ROOT=$JOB_ROOT"
echo "DATASET_NAME=$DATASET_NAME"
echo "NPZ=$NPZ"
echo "WORKERS=$WORKERS"

echo "=== [1/2] raw EEG -> 58-participant FAST-compatible NPZ ==="
python \
  "$WILSON/benchmark_eeg_preprocess.py" \
  --raw-root "${RAW_ROOT:-/path/to/raw_standalone_eeg}" \
  --montage "${MONTAGE:-/path/to/EasyCap/BC-MR-64.bvef}" \
  --out-root "$OUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --eog-threshold 3.5 \
  --muscle-topk 10 \
  --workers "$WORKERS" \
  "${FORCE_ARGS[@]}"

echo "=== [2/2] FAST zip-parameter training on 58-subject NPZ ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
  "$WILSON/run_fast_zipparams_npz.py" \
  --repo-root "$ZIP_REPRO" \
  --import-dir "$WILSON/fast_58_imports" \
  --run-cwd "$RUN_CWD" \
  --train-fast "$ZIP_REPRO/train_FAST.py" \
  --npz "$NPZ" \
  --dataset-name "$DATASET_NAME" \
  --gpu 0 \
  --folds 0,1,2,3,4,5,6,7,8,9 \
  --bs 200 \
  --n-subjects 58

echo "END $(date '+%F %T %z')"
