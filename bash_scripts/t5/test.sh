#!/bin/bash

TIME=0
MEM=30000
GRES=gpu:A6000:1
CPUS=8

LOG=/projects/tir6/general/vpratapa/research/slurm-logs/ts-dev
mkdir -p $LOG

if [[ -z $EXCLUDE ]]; then
    echo "no nodes were excluded"
    exit
fi

CONFIG=configs/test.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --cpus-per-task $CPUS \
    --gres $GRES \
    --exclude $EXCLUDE \
    --output $LOG/hf-eval-test-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/t5/eval.slurm
