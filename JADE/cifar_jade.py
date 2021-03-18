import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --start_lambda 0.01 --end_lambda 0.0192308",
        "python GNN_training/cross_validation.py --start_lambda 0.0215385 --end_lambda 0.0307692",
        "python GNN_training/cross_validation.py --start_lambda 0.0330769 --end_lambda 0.0423077",
        "python GNN_training/cross_validation.py --start_lambda 0.0446154 --end_lambda 0.0538462",
        "python GNN_training/cross_validation.py --start_lambda 0.0561539 --end_lambda 0.0653846",
        "python GNN_training/cross_validation.py --start_lambda 0.0676923 --end_lambda 0.0769231",
        "python GNN_training/cross_validation.py --start_lambda 0.0792308 --end_lambda 0.0884615",
        "python GNN_training/cross_validation.py --start_lambda 0.0907692 --end_lambda 0.1"]


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
