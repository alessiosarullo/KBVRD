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
OUTPUT_DIR="output/${NET}"
DATETIME=`date +'%Y-%m-%d_%H-%M-%S'`
EXP_FULL_NAME="${DATETIME}_${EXP_NAME}"
EXP_DIR=${OUTPUT_DIR}/${EXP_FULL_NAME}
LOG="$EXP_DIR/log.txt"
LAST_EXP_SYMLINK="output/last_exp"

mkdir -p ${EXP_DIR}
rm -f -- ${LAST_EXP_SYMLINK}
ln -rs ${EXP_DIR} ${LAST_EXP_SYMLINK}
exec &> >(tee -a "$LOG")
echo Logging ${EXP_DIR} to "$LOG"

python -u scripts/run.py --model ${NET} --save_dir ${EXP_FULL_NAME} "${@:4}"