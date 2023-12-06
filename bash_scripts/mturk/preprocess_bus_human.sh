#!/bin/bash

python src/mturk/prepare_background_questions.py \
    --gold data \
    --out outputs/mturk/question_data \
    --split dev test
