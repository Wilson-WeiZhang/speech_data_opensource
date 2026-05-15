#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/benchmark_work}"
PROJECT="${PROJECT:-/path/to/speech_data_opensource}"
SCRIPT="$PROJECT/benchmark_fast_reproduction/preprocess/benchmark_eeg_preprocess.py"
OUT_ROOT="$ROOT/wilson/fast_benchmark_58"
DATASET="${DATASET:-fast_benchmark_58}"
LOG_DIR="$ROOT/wilson/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/fast_benchmark_st_preprocess_${STAMP}.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/fast_benchmark_st_preprocess_latest.log"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"

{
  echo "START $(date -Is)"
  echo "SCRIPT=$SCRIPT"
  echo "OUT_ROOT=$OUT_ROOT"
  echo "DATASET=$DATASET"
  nice -n 10 python "$SCRIPT" \
    --dataset st \
    --raw-root "${RAW_ROOT:-/path/to/raw_standalone_eeg}" \
    --montage "${MONTAGE:-/path/to/EasyCap/BC-MR-64.bvef}" \
    --out-root "$OUT_ROOT" \
    --dataset-name "$DATASET" \
    --expected-subjects 58 \
    --expected-trials 100 \
    --strict-balanced \
    --workers "${BENCHMARK_PREP_WORKERS:-6}"
  echo "END $(date -Is)"
} 2>&1 | tee "$LOG"
