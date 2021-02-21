#!/bin/bash

exp_num=$SLURM_ARRAY_TASK_ID

module purge
module load cuda/10.1

export PATH=$PATH:/jmain01/home/JAD035/pkm01/axa50-pkm01/miniconda3/bin
source activate /jmain01/home/JAD035/pkm01/axa50-pkm01/miniconda3/envs/GNN_Attention_Mechanism

grbgetkey --path /jmain01/home/JAD035/pkm01/axa50-pkm01/gurobi911/ 07fd97ea-742b-11eb-932a-020d093b5256
export GRB_LICENSE_FILE=/jmain01/home/JAD035/pkm01/axa50-pkm01/gurobi911/gurobi.lic

nvidia-smi
nvcc --version
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python JADE/cifar_jade.py --exp_num $exp_num