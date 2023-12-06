#!/bin/bash

TIME=5-00:00:00
MEM=120000
GRES=gpu:A6000:2
CPUS=8

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/train.conf
CONFIG_NAME=$1
MASTER_PORT=$2

sbatch \
    --time $TIME \
    --mem $MEM \
    --cpus-per-task $CPUS \
    --gres $GRES \
    --exclude $EXCLUDE \
    --output $LOG/hf-train-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME},MASTER_PORT=${MASTER_PORT} \
    bash_scripts/t5/train.slurm
