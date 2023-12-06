#!/bin/bash

TIME=24:00:00
MEM=30000
GRES=gpu:1

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/test.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --gres $GRES \
    --output $LOG/score-test-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/t5/score.slurm
