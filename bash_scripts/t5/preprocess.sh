#!/bin/bash

TIME=2:00:00
MEM=5000

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/train.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --output $LOG/preprocess-hf-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/t5/preprocess.slurm
