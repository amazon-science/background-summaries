#!/bin/bash

TIME=48:00:00
MEM=7000

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/gpt.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --output $LOG/predict-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/gpt/sbatch_predict.sh
