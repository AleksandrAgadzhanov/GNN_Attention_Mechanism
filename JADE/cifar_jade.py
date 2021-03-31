import os
import time
import torch as th
import argparse
import sys


# jobs = ["python GNN_training/cross_validation.py --start_lambda 0.02 --end_lambda 0.026 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.027 --end_lambda 0.033 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.034 --end_lambda 0.04 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.041 --end_lambda 0.047 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.048 --end_lambda 0.054 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.055 --end_lambda 0.061 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.062 --end_lambda 0.068 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.069 --end_lambda 0.075 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.076 --end_lambda 0.082 --num 7",
#         "python GNN_training/cross_validation.py --start_lambda 0.083 --end_lambda 0.09 --num 8"]
jobs = ["python project_motivation/pgd_attacks.py"]


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
