import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import time
import torch
import torch.nn as nn
from replay_buffer import make_replay_loader_dist
from libero.libero import benchmark
from torch.distributed import init_process_group, destroy_process_group, gather
from pathlib import Path
from tqdm import tqdm
import utils.misc as utils

import pickle
import pprint
from libero.libero import benchmark
import numpy as np

torch.backends.cudnn.benchmark = True


task_num = 10
result_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num_eval_episodes = 20
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_goal"]()

if __name__ == "__main__":
    with open("result_pickle.txt", 'w') as f:
        for j in range(task_num):
            this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
            print("this_result_list :", this_result_list)
            this_result_list = np.array(this_result_list)
            avg_success = np.mean(this_result_list, axis=0)
            task = task_suite.get_task(j)
            task_name = task.name
            print(f"Success rates for task {j} {task_name}:")
            print(f"{avg_success * 100:.1f}%")

            f.write(f"Success rates for task {j} {task_name}:")
            f.write(f"{avg_success * 100:.1f}%\n")