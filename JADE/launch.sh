#!/bin/bash

exp_num=$SLURM_ARRAY_TASK_ID
module purge

module load cuda/10.1
source activate /jmain01/home/JAD035/pkm01/fxj15-pkm01/anaconda3/envs/python_38

export VISION_DATA="/jmain01/home/JAD035/pkm01/fxj15-pkm01/data/"

nvidia-smi
nvcc --version
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python jade/cifar_jade.py --exp_num $exp_num
