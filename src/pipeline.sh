#!/bin/bash
set -e
###########################################
EXP_NAME=$1
BATCH_SIZE=1
###########################################
### Learning rate

LR=(0.001 0.0007 0.0003 0.0001 )
echo Learning Rates ${LR[@]}
###########################################

echo $(date) Train stage 0
echo python train.py -e ${EXP_NAME}_stage0 -bs ${BATCH_SIZE} -st 0 -lr ${LR[0]}
python train.py -e ${EXP_NAME}_stage0 -bs ${BATCH_SIZE} -st 0 -lr ${LR[0]}

# loop 1 to 3 (3 included)
for i in {1..3}
do
echo -e "\n\n"
echo $(date) Generate Data for stage $i from previous stage
echo python data_gen.py -e ${EXP_NAME}_stage$((i-1)) -st ${i}
python data_gen.py -e ${EXP_NAME}_stage$((i-1)) -st ${i}

echo -e "\n"
echo $(date) Train stage $i
echo python train.py -e ${EXP_NAME}_stage${i} -bs ${BATCH_SIZE} -st ${i} -lr ${LR[${i}]}
python train.py -e ${EXP_NAME}_stage${i} -bs ${BATCH_SIZE} -st ${i} -lr ${LR[${i}]}

done

