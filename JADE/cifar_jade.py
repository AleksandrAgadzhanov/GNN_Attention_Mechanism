import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/training_dataset_generation.py --start_index 0 --end_index 409",
        "python GNN_training/training_dataset_generation.py --start_index 409 --end_index 818",
        "python GNN_training/training_dataset_generation.py --start_index 818 --end_index 1227",
        "python GNN_training/training_dataset_generation.py --start_index 1227 --end_index 1636",
        "python GNN_training/training_dataset_generation.py --start_index 1636 --end_index 2045",
        "python GNN_training/training_dataset_generation.py --start_index 2045 --end_index 2454",
        "python GNN_training/training_dataset_generation.py --start_index 2454 --end_index 2863",
        "python GNN_training/training_dataset_generation.py --start_index 2863 --end_index 3272",
        "python GNN_training/training_dataset_generation.py --start_index 3272 --end_index 3681",
        "python GNN_training/training_dataset_generation.py --start_index 3681 --end_index 4085"]


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
