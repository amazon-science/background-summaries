#!/bin/bash

module load cuda-11.8

accelerate launch \
    src/eval.py \
    --config $CONFIG \
    --config-name $CONFIG_NAME
