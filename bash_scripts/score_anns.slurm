#!/bin/bash

#SBATCH --time 15:00:00
#SBATCH --mem 10000
#SBATCH --gres gpu:1
#SBATCH --output slurm-logs/score-anns-%j.out

OUT=outputs/anns

export PYTHONPATH="$PYTHONPATH:src/bus"

for BUS_CONFIG_NAME in gpt-3.5-turbo gpt-4-0613; do
    for SPLIT in dev test; do
        python src/score_anns.py \
            data \
            $SPLIT \
            $OUT \
            --bus-config-name $BUS_CONFIG_NAME
    done
done
