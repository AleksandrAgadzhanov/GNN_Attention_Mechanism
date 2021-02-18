import os
import time
import torch as th
import argparse
import sys


# # generating difficult SAT datasets
# jobs = ["python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name easy",
#         "python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name wide",
#         "python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name deep",
#         "python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name train",
#         "python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name val"]
#
# # training GNNs on various very large SAT/UNSAT datasets
# jobs = ["python adv_exp/scripts/script_training_jade.py --exp_name train_table_eps03_jade",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_table_eps03_n4e3",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_table_eps03_n4e4",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_table_eps02_n4e3",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_table_eps025_n4e3"]
#
# # generating easy but large train SAT dataset
# jobs = ["python adv_exp/scripts/script_create_dataset_jade.py --jade --exp_name train_quick"]
#
# # run on new larger train dataset
# jobs = ["python adv_exp/scripts/script_training_jade.py --exp_name train_jade_new_train_n25e4"]
#
# # train 1. more steps, 2. with momentum, 3. with adam, 4. with prop lr decay
# jobs = ["python adv_exp/scripts/script_training_jade.py --exp_name train_jade_n25e4_horizon40",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_n25e4_momentum_01",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_n25e4_adam",
#         "python adv_exp/scripts/script_training_jade.py --exp_name train_jade_n25e4_rel_decay"]

jobs = ["python GNN_training/cross_validation.py"]


def run_command(command, noprint=True):
    command = " ".join(command.split())
    print(command)
    os.system(command)


def launch(jobs, interval):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job)
        time.sleep(interval)


if __name__ == "__main__":
    print(sys.version_info)
    print(th.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', type=int)
    args = parser.parse_args()
    print(f'jobs before{jobs}')
    jobs = [jobs[args.exp_num]]
    print(f'jobs after{jobs}')

    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)
