import os
import time
import torch as th
import argparse
import sys

# jobs = ["python GNN_training/train_GNN.py",
#         "python GNN_training/cross_validation.py --start_lambda 0.01 --end_lambda 0.02 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.03 --end_lambda 0.04 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.05 --end_lambda 0.06 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.07 --end_lambda 0.08 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.09 --end_lambda 0.1 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.2 --end_lambda 0.3 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.4 --end_lambda 0.5 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.6 --end_lambda 0.7 --num 2",
#         "python GNN_training/cross_validation.py --start_lambda 0.8 --end_lambda 1.0 --num 3"]


jobs = ["python GNN_framework/attack_properties_with_pgd.py --filename base_easy_SAT_jade",
        "python GNN_framework/attack_properties_with_pgd.py --filename easy_base_easy_SAT_jade",]
        # "python GNN_training/cross_validation.py --start_lambda 0.031 --end_lambda 0.035 --num 5",
        # "python GNN_training/cross_validation.py --start_lambda 0.036 --end_lambda 0.039 --num 4",
        # "python GNN_training/cross_validation.py --start_lambda 0.041 --end_lambda 0.045 --num 5",
        # "python GNN_training/cross_validation.py --start_lambda 0.046 --end_lambda 0.049 --num 4",
        # "python GNN_training/cross_validation.py --start_lambda 0.051 --end_lambda 0.055 --num 5",
        # "python GNN_training/cross_validation.py --start_lambda 0.056 --end_lambda 0.059 --num 4",
        # "python GNN_training/cross_validation.py --start_lambda 0.061 --end_lambda 0.065 --num 5",
        # "python GNN_training/cross_validation.py --start_lambda 0.066 --end_lambda 0.069 --num 4"]


def run_command(command):
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
