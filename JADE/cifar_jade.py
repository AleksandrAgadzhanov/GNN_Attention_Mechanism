import os
import time
import torch as th
import argparse
import sys

jobs = ["python GNN_training/cross_validation.py --start_lambda 0.043 --end_lambda 0.043 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 0.044 --end_lambda 0.044 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 0.045 --end_lambda 0.045 --num 1",
        "python GNN_framework/attack_properties_with_pgd.py --filename base_easy_SAT_jade",
        "python GNN_framework/attack_properties_with_pgd.py --filename easy_base_easy_SAT_jade"]


# jobs = ["python GNN_training/cross_validation.py --start_lambda 0.031 --end_lambda 0.035 --num 5",
#         "python GNN_training/cross_validation.py --start_lambda 0.036 --end_lambda 0.039 --num 4",
#         "python GNN_training/cross_validation.py --start_lambda 0.041 --end_lambda 0.045 --num 5",
#         "python GNN_training/cross_validation.py --start_lambda 0.046 --end_lambda 0.049 --num 4",
#         "python GNN_training/cross_validation.py --start_lambda 0.051 --end_lambda 0.055 --num 5",
#         "python GNN_training/cross_validation.py --start_lambda 0.056 --end_lambda 0.059 --num 4",
#         "python GNN_training/cross_validation.py --start_lambda 0.061 --end_lambda 0.065 --num 5",
#         "python GNN_training/cross_validation.py --start_lambda 0.066 --end_lambda 0.069 --num 4"]


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
