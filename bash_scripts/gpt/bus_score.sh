#!/bin/bash

TIME=24:00:00
MEM=7000

LOG=slurm-logs
mkdir -p $LOG

CONFIG=configs/gpt.conf
CONFIG_NAME=$1
BUS_CONFIG=configs/bus.conf
BUS_CONFIG_NAME=$2

sbatch \
    --time $TIME \
    --mem $MEM \
    --output $LOG/score-bus-${CONFIG_NAME}-${BUS_CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME},BUS_CONFIG=${BUS_CONFIG},BUS_CONFIG_NAME=${BUS_CONFIG_NAME} \
    bash_scripts/gpt/b_score.slurm
