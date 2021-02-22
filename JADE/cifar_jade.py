import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/training_dataset_generation.py --start_index 0 --end_index 15",
        "python GNN_training/training_dataset_generation.py --start_index 15 --end_index 30",
        "python GNN_training/training_dataset_generation.py --start_index 30 --end_index 45",
        "python GNN_training/training_dataset_generation.py --start_index 45 --end_index 60",
        "python GNN_training/training_dataset_generation.py --start_index 60 --end_index 75",
        "python GNN_training/training_dataset_generation.py --start_index 75 --end_index 90",
        "python GNN_training/training_dataset_generation.py --start_index 90 --end_index 115",
        "python GNN_training/training_dataset_generation.py --start_index 115 --end_index 135"]


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
