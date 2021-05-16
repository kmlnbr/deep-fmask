#!/bin/bash

## This script implements the 4 stage self-training pipeline.


# Exit scrpt if any command fails.
set -e

## Pipeline Arguments

# The name of the experiment is the prefix used in naming each stage. For example, if the experiment
# name is defined by the user as exp1, then the complete experiment name for stage 0 will be
# exp1_stage0
EXP_NAME=$1

# Set the batch size depending on the memory capacity of the GPU.
BATCH_SIZE=1

# Define the learning rate for each stage
LR=(0.001 0.0007 0.0003 0.0001 )

## Pipeline Commands
echo $(date) Stage 0 : Train model
echo python train.py -e ${EXP_NAME}_stage0 -bs ${BATCH_SIZE} -st 0 -lr ${LR[0]}
python train.py -e ${EXP_NAME}_stage0 -bs ${BATCH_SIZE} -st 0 -lr ${LR[0]}

for i in {1..3}
do
echo -e "\n\n"
echo $(date) Generate Data for stage $i using model trained in previous stage
echo python data_gen.py -e ${EXP_NAME}_stage$((i-1)) -st ${i}
python data_gen.py -e ${EXP_NAME}_stage$((i-1)) -st ${i}

echo -e "\n"
echo $(date) Stage $i : Train model
echo python train.py -e ${EXP_NAME}_stage${i} -bs ${BATCH_SIZE} -st ${i} -lr ${LR[${i}]}
python train.py -e ${EXP_NAME}_stage${i} -bs ${BATCH_SIZE} -st ${i} -lr ${LR[${i}]}

done

