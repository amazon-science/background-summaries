#!/bin/bash

module load gcc-7.4 cuda-11.8

TRANSFORMERS_NO_ADVISORY_WARNINGS=1

deepspeed \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --config $CONFIG \
    --config-name $CONFIG_NAME
