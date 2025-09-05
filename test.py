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

torch.backends.cudnn.benchmark = True

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode
    
def construct_task_data_path(root_dir, task_name, task_data_dir_suffix='framestack1'):
    return Path(root_dir) / (task_name.lower()+('' if not task_data_dir_suffix or task_data_dir_suffix == 'None' else task_data_dir_suffix))

def replay_iter(replay_loader):
    return iter(replay_loader)

pretraining_data_dirs = []
for task_id in range(90): 
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_90']()
    task = task_suite.get_task(task_id)
    task_name = task.name
    offline_data_dir = construct_task_data_path("/home/dls0901/data_prise/prise", task_name, "_framestack1")
    pretraining_data_dirs.append(offline_data_dir)
eval_env = None

last_action_set = {-1:0, 0:0, 1:0}
epi_lens = []

eps_fns = []
for replay_dir in pretraining_data_dirs:
    eps_fns.extend(utils.choose(sorted(replay_dir.glob('*.npz'), reverse=True), 1000000))

for eps_idx, eps_fn in tqdm(enumerate(eps_fns)):
    epi = load_episode(eps_fn)
    epi_len = episode_len(epi)
    epi_lens.append(epi_len)

    for i in range(epi_len):
        # action = epi['action'][i].astype(np.float32)
        # last_action_set[action[-1]] += 1
        print(epi['action'][i])
        break
    break

# print(last_action_set)
# print(np.sum(epi_lens))
# print(np.mean(epi_lens))
