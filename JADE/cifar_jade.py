import os
import time
import torch as th
import argparse
import sys


jobs = ["python GNN_training/training_dataset_generation.py --start_index 0 --end_index 50",
        "python GNN_training/training_dataset_generation.py --start_index 50 --end_index 100",
        "python GNN_training/training_dataset_generation.py --start_index 100 --end_index 150",
        "python GNN_training/training_dataset_generation.py --start_index 150 --end_index 200",
        "python GNN_training/training_dataset_generation.py --start_index 200 --end_index 250",
        "python GNN_training/training_dataset_generation.py --start_index 250 --end_index 300",
        "python GNN_training/training_dataset_generation.py --start_index 300 --end_index 350",
        "python GNN_training/training_dataset_generation.py --start_index 350 --end_index 400",
        "python GNN_training/training_dataset_generation.py --start_index 400 --end_index 430"]


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
