#!/bin/bash

set -e


EXP_PRE=$1

echo -e "\n\n"
for i in {0..3}                                                                                                                                                                                                
do 
EXP_NAME=${EXP_PRE}_stage${i}
echo $(date) Running Prediction Data
echo python predict.py -e ${EXP_NAME} -p /home/nambiar/Downloads/new_s2/PREDICT/P1
python predict.py -e ${EXP_NAME} -p /home/nambiar/Downloads/new_s2/PREDICT/P1

echo -e "\n\n"
echo $(date) Running Validation Data
echo python predict.py -e ${EXP_NAME} -p /home/nambiar/Downloads/new_s2/PREDICT/V1
python predict.py -e ${EXP_NAME} -p /home/nambiar/Downloads/new_s2/PREDICT/V1
done
