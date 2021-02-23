import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --loss_lambda 0.0",
        "python GNN_training/cross_validation.py --loss_lambda 0.001",
        "python GNN_training/cross_validation.py --loss_lambda 0.01",
        "python GNN_training/cross_validation.py --loss_lambda 0.1",
        "python GNN_training/cross_validation.py --loss_lambda 1.0",
        "python GNN_training/cross_validation.py --loss_lambda 10.0"]


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
