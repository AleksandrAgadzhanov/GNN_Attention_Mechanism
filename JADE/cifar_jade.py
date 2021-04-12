import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/train_GNN.py",
        "python GNN_training/cross_validation.py --start_lambda 0.6 --end_lambda 0.6 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 0.7 --end_lambda 0.7 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 0.8 --end_lambda 0.8 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 0.9 --end_lambda 0.9 --num 1",
        "python GNN_training/cross_validation.py --start_lambda 1.0 --end_lambda 1.0 --num 1"]


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
