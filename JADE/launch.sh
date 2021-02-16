#!/bin/bash

exp_num=$SLURM_ARRAY_TASK_ID
module purge
# module load python3/anaconda
module load cuda/10.1
export PATH=$PATH:/jmain01/home/JAD035/pkm01/axa50-pkm01/GNN_Attention_Mechanism
source activate /jmain01/home/JAD035/pkm01/axa50-pkm01/miniconda3/venv/Scripts
# source activate /jmain01/home/JAD035/pkm01/axa50-pkm01/anaconda3/envs/python_38
# export VISION_DATA="/jmain01/home/JAD035/pkm01/fxj15-pkm01/data/"
# export PATH=$PATH:/jmain01/home/JAD035/pkm01/axa50-pkm01/GNN_Attention_Mechanism
# echo 'export PATH=$PATH:/jmain01/home/JAD035/pkm01/fxj15-pkm01/workspace/Verification_bab'  >> ~/.bash_profile

# echo "path variables" $PATH

nvidia-smi
nvcc --version
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python JADE/cifar_jade.py --exp_num $exp_num
# python cifar_jade.py