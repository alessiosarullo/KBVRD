#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET='base'
EXP_NAME=$1
GPU_ID=$2
# The following parameters are optional: a default value is provided and it is only substituted if the relative argument is unset or has a null value
# (e.g., the empty string ''). Remove the colon to only substitute if unset.

export CUDA_VISIBLE_DEVICES=$GPU_ID

# log
CHECKPOINT_DIR="output/${NET}"
DATETIME=`date +'%Y-%m-%d_%H-%M-%S'`
EXP_FULL_NAME="${DATETIME}_${EXP_NAME}"
EXP_DIR=${CHECKPOINT_DIR}/${EXP_FULL_NAME}
LOG="$EXP_DIR/log.txt"

mkdir -p ${EXP_DIR}
echo Logging to "$LOG"

python -u scripts/run.py --save_dir ${EXP_DIR} "${@:3}" >${LOG} 2>&1



