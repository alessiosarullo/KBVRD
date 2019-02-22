#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
EXP_NAME=$2
GPU_ID=$3
# The following parameters are optional: a default value is provided and it is only substituted if the relative argument is unset or has a null value
# (e.g., the empty string ''). Remove the colon to only substitute if unset.

export CUDA_VISIBLE_DEVICES=$GPU_ID

# log
CHECKPOINT_DIR="checkpoints/${NET}"
DATETIME=`date +'%Y-%m-%d_%H-%M-%S'`
EXP_FULL_NAME="${DATETIME}_${EXP_NAME}"
EXP_DIR=$CHECKPOINT_DIR/$EXP_FULL_NAME
LOG="$EXP_DIR/log.txt"

mkdir -p ${EXP_DIR}
echo Logging to "$LOG"

python -u scripts/run.py --save_dir ${EXP_DIR} "${@:4}" >${LOG} 2>&1

#./scripts/eval.sh ${EXP_DIR} ${GPU_ID} 0 "${@:4}"



