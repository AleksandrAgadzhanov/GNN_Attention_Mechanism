import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/cross_validation.py --loss_lambda 0.1",
        "python GNN_training/cross_validation.py --loss_lambda 0.111",
        "python GNN_training/cross_validation.py --loss_lambda 0.124",
        "python GNN_training/cross_validation.py --loss_lambda 0.138",
        "python GNN_training/cross_validation.py --loss_lambda 0.154",
        "python GNN_training/cross_validation.py --loss_lambda 0.172",
        "python GNN_training/cross_validation.py --loss_lambda 0.191",
        "python GNN_training/cross_validation.py --loss_lambda 0.213",
        "python GNN_training/cross_validation.py --loss_lambda 0.238",
        "python GNN_training/cross_validation.py --loss_lambda 0.265",
        "python GNN_training/cross_validation.py --loss_lambda 0.295",
        "python GNN_training/cross_validation.py --loss_lambda 0.329",
        "python GNN_training/cross_validation.py --loss_lambda 0.366",
        "python GNN_training/cross_validation.py --loss_lambda 0.408",
        "python GNN_training/cross_validation.py --loss_lambda 0.454",
        "python GNN_training/cross_validation.py --loss_lambda 0.506",
        "python GNN_training/cross_validation.py --loss_lambda 0.564",
        "python GNN_training/cross_validation.py --loss_lambda 0.629",
        "python GNN_training/cross_validation.py --loss_lambda 0.700",
        "python GNN_training/cross_validation.py --loss_lambda 0.780",
        "python GNN_training/cross_validation.py --loss_lambda 0.869",
        "python GNN_training/cross_validation.py --loss_lambda 0.969",
        "python GNN_training/cross_validation.py --loss_lambda 1.08",
        "python GNN_training/cross_validation.py --loss_lambda 1.2",
        "python GNN_training/cross_validation.py --loss_lambda 1.34",
        "python GNN_training/cross_validation.py --loss_lambda 1.49",
        "python GNN_training/cross_validation.py --loss_lambda 1.66",
        "python GNN_training/cross_validation.py --loss_lambda 1.85",
        "python GNN_training/cross_validation.py --loss_lambda 2.06",
        "python GNN_training/cross_validation.py --loss_lambda 2.3",
        "python GNN_training/cross_validation.py --loss_lambda 2.56",
        "python GNN_training/cross_validation.py --loss_lambda 2.86",
        "python GNN_training/cross_validation.py --loss_lambda 3.18",
        "python GNN_training/cross_validation.py --loss_lambda 3.55",
        "python GNN_training/cross_validation.py --loss_lambda 3.95",
        "python GNN_training/cross_validation.py --loss_lambda 4.4",
        "python GNN_training/cross_validation.py --loss_lambda 4.9",
        "python GNN_training/cross_validation.py --loss_lambda 5.46",
        "python GNN_training/cross_validation.py --loss_lambda 6.09",
        "python GNN_training/cross_validation.py --loss_lambda 6.78",
        "python GNN_training/cross_validation.py --loss_lambda 7.56",
        "python GNN_training/cross_validation.py --loss_lambda 8.42",
        "python GNN_training/cross_validation.py --loss_lambda 9.38",
        "python GNN_training/cross_validation.py --loss_lambda 10.45",
        "python GNN_training/cross_validation.py --loss_lambda 11.65",
        "python GNN_training/cross_validation.py --loss_lambda 12.98",
        "python GNN_training/cross_validation.py --loss_lambda 14.46",
        "python GNN_training/cross_validation.py --loss_lambda 16.11",
        "python GNN_training/cross_validation.py --loss_lambda 17.95",
        "python GNN_training/cross_validation.py --loss_lambda 20.0"]


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
