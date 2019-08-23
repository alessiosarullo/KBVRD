#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
EXP_NAME=$2
GPU_ID=$3
NUM_RUNS=$4

export CUDA_VISIBLE_DEVICES=$GPU_ID
OUTPUT_DIR="output/${NET}"

EXPS=()
for IDX in $(seq 1 "${NUM_RUNS}")
do
  DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
  EXP_FULL_NAME="${DATETIME}_${EXP_NAME}_RUN${IDX}"
  EXP_DIR=${OUTPUT_DIR}/${EXP_FULL_NAME}
  LOG="$EXP_DIR/log.txt"

  EXPS+=("${EXP_DIR}")
  mkdir -p "${EXP_DIR}"
  echo Logging "${EXP_DIR}" to "$LOG"

  python -u scripts/run.py --model "${NET}" --save_dir "${EXP_FULL_NAME}" --randomize "${@:5}" > "${LOG}" 2>&1
done
DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
python -u scripts/aggregate_tb_runs.py "${OUTPUT_DIR}/${DATETIME}_${EXP_NAME}_AGGR${NUM_RUNS}" "${EXPS[@]}"