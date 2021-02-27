import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --start_lambda 0.001 --end_lambda 0.01 --num 9 --endpoint False",
        "python GNN_training/cross_validation.py --start_lambda 0.01 --end_lambda 0.1 --num 9 --endpoint False"
        "python GNN_training/cross_validation.py --start_lambda 0.1 --end_lambda 1 --num 9 --endpoint False"
        "python GNN_training/cross_validation.py --start_lambda 1 --end_lambda 10 --num 9 --endpoint False"
        "python GNN_training/cross_validation.py --start_lambda 10 --end_lambda 100 --num 10 --endpoint True"]


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
