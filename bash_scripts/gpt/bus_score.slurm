#!/bin/bash

python src/bus/score.py \
    --config $CONFIG \
    --config-name ${CONFIG_NAME} \
    --bus-config ${BUS_CONFIG} \
    --bus-config-name ${BUS_CONFIG_NAME}
