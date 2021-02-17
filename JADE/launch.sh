#!/bin/bash
echo hostid
exp_num=$SLURM_ARRAY_TASK_ID

module purge

module load cuda/10.1

export PATH=$PATH:/jmain01/home/JAD035/pkm01/axa50-pkm01/miniconda3/bin

module load Gurobi/9.1.1
export GRB_LICENSE_FILE=/jmain01/home/JAD035/pkm01/axa50-pkm01/gurobi911/gurobi.lic

source activate /jmain01/home/JAD035/pkm01/axa50-pkm01/miniconda3/envs/GNN_Attention_Mechanism

nvidia-smi
nvcc --version
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python JADE/cifar_jade.py --exp_num $exp_num