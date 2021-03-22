import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --start_lambda 0.03 --end_lambda 0.0371795",
        "python GNN_training/cross_validation.py --start_lambda 0.0389744 --end_lambda 0.0461539",
        "python GNN_training/cross_validation.py --start_lambda 0.0479487 --end_lambda 0.0551282",
        "python GNN_training/cross_validation.py --start_lambda 0.0569231 --end_lambda 0.0641026",
        "python GNN_training/cross_validation.py --start_lambda 0.0658974 --end_lambda 0.0730769",
        "python GNN_training/cross_validation.py --start_lambda 0.0748718 --end_lambda 0.0820513",
        "python GNN_training/cross_validation.py --start_lambda 0.0838462 --end_lambda 0.0910256",
        "python GNN_training/cross_validation.py --start_lambda 0.0928205 --end_lambda 0.1"]


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
