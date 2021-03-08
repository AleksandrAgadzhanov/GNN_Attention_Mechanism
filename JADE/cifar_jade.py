import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --start_lambda 0.045 --end_lambda 0.0557692",
        "python GNN_training/cross_validation.py --start_lambda 0.0584615 --end_lambda 0.0692308",
        "python GNN_training/cross_validation.py --start_lambda 0.0719231 --end_lambda 0.0826923",
        "python GNN_training/cross_validation.py --start_lambda 0.0853846 --end_lambda 0.0961539",
        "python GNN_training/cross_validation.py --start_lambda 0.0988462 --end_lambda 0.109615",
        "python GNN_training/cross_validation.py --start_lambda 0.112308 --end_lambda 0.123077",
        "python GNN_training/cross_validation.py --start_lambda 0.125769 --end_lambda 0.136538",
        "python GNN_training/cross_validation.py --start_lambda 0.139231 --end_lambda 0.15"]


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
