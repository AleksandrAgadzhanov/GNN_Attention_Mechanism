#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=2:00:00

# set name of job
#SBATCH --job-name=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=aleksandr.agadzhanov@st-hildas.ox.ac.uk

chmod u+x JADE/launch.sh
JADE/launch.sh 0