#!/bin/bash

python src/mturk/prepare_data.py \
    --config configs/eval.conf configs/gpt.conf \
    --config-name flan-t5-xl gpt-3.5-turbo \
    --gold data \
    --out outputs/mturk/data \
    --split dev test
