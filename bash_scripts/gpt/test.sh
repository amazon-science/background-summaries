#!/bin/bash

TIME=48:00:00
MEM=7000

LOG=/projects/tir6/general/vpratapa/research/slurm-logs/ts-dev
mkdir -p $LOG

if [[ -z $EXCLUDE ]]; then
    echo "no nodes were excluded"
    exit
fi

CONFIG=configs/gpt-test.conf
CONFIG_NAME=$1

sbatch \
    --time $TIME \
    --mem $MEM \
    --exclude $EXCLUDE \
    --output $LOG/predict-${CONFIG_NAME}-%j.out \
    --export ALL,CONFIG=$CONFIG,CONFIG_NAME=${CONFIG_NAME} \
    bash_scripts/gpt/sbatch_predict.sh
