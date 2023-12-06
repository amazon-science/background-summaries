#!/bin/bash

TIME=10:00:00
MEM=10000
GRES=gpu:1

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/gpt.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --gres $GRES \
    --output $LOG/score-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/gpt/score.slurm
