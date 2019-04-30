#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1
GPU_ID=$2

export CUDA_VISIBLE_DEVICES=$GPU_ID

# log
OUTPUT_DIR="pyGAT/output"
DATETIME=`date +'%Y-%m-%d_%H-%M-%S'`
EXP_FULL_NAME="${DATETIME}_${EXP_NAME}"
EXP_DIR=${OUTPUT_DIR}/${EXP_FULL_NAME}
LOG="$EXP_DIR/log.txt"

mkdir -p ${EXP_DIR}
exec &> >(tee -a "$LOG")
echo Logging ${EXP_DIR} to "$LOG"

python -u pyGAT/train.py --save_dir ${EXP_DIR} "${@:3}"