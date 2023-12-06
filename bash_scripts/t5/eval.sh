#!/bin/bash

TIME=0
MEM=30000
GRES=gpu:A6000:1
CPUS=8

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/eval.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --cpus-per-task $CPUS \
    --gres $GRES \
    --output $LOG/hf-eval-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/t5/eval.slurm
