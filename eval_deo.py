import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import copy
import pickle
import io
import distutils.dir_util
import hydra
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group, gather
from libero.libero import benchmark
import utils.misc as utils
import utils.libero_wrapper as libero_wrapper
from utils.logger import Logger
from replay_buffer import make_replay_loader_dist
import torch.multiprocessing as mp
from collections import defaultdict, deque
from tokenizer_api import Tokenizer
from tqdm import tqdm

from utils.libero_dataset import LiberoGoalDataset

from de import *

torch.backends.cudnn.benchmark = True
SEED = 0

# Fix Random Seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(SEED)

#################### ARGUMENTS #####################
DEVICE = 'cuda:0'
DOWNSTREAM_TASK_SUITE = 'libero_goal'
NUM_EVAL_EPI = 20
EVAL_MAX_STEP = 600

ENCODER_PATH = "pretrained_model/encoder_16.pt"

RESULT_FILE_NAME = "performance_deO.pkl"
AGENT_PATH = "/data/libero/exp_results/deo.pt"
#################### ARGUMENTS #####################

dataset = LiberoGoalDataset()

task_idx = dict()
for i, t in enumerate(dataset.task_names):
    task_idx[str(t).split('/')[-1]] = i
print(task_idx)


for i in tqdm(range(10)):
    agent = MYMODEL(device=DEVICE).to(DEVICE)

    agent.train(False)
    # agent.decoder_head.training = False
    # agent.decoder_head.low_eval_noise = True
    
    # agent.a_quantizer.to(DEVICE)
    eval_until_episode = utils.Until(NUM_EVAL_EPI)
    # num_tasks_per_gpu = (90//self.world_size) + 1

    print(AGENT_PATH)
    agent = torch.load(AGENT_PATH).to(DEVICE)
    
    WINDOW_SIZE = 10

    ### Setup eval environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[DOWNSTREAM_TASK_SUITE]()
    task = task_suite.get_task(i)
    eval_env = libero_wrapper.make(i, DOWNSTREAM_TASK_SUITE, 
                                    img_size=224,
                                    seed=SEED)
    eval_env.task_name = task.name
    eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)
    task_name = task.name
    
    eval_start_time = time.time()
    eval_until_episode = utils.Until(NUM_EVAL_EPI + WINDOW_SIZE)

    counter, episode, success = 0, 0, 0
    while eval_until_episode(episode):
        time_step = eval_env.reset()
        step, cur_goal = 0, None
        obs_total_list = []
        while step < EVAL_MAX_STEP:
            if time_step['done']:
                success += 1
                break
            with torch.no_grad():
                obs_agent = time_step.agentview
                task_embedding = eval_env.task_embedding
                obs_wrist = time_step.wristview
                state     = time_step.state 
                
                ### Encode the current timestep
                task_embedding = torch.torch.as_tensor(task_embedding, device=DEVICE)
                obs_agent = torch.torch.as_tensor(obs_agent.copy(), device=DEVICE).unsqueeze(0)
                obs_wrist = torch.torch.as_tensor(obs_wrist.copy(), device=DEVICE).unsqueeze(0)
                state     = torch.torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                goal_obs = dataset.goals[task_idx[task_name]].to(DEVICE)

                n_step, num_channel, img_size = obs_agent.shape[0], obs_agent.shape[1], obs_agent.shape[2]
                
                obs_embed = obs_agent.reshape(-1, num_channel, img_size, img_size).float() / 255.
                obs_wrist = obs_wrist.reshape(-1, num_channel, img_size, img_size).float() / 255.

                obs_total = torch.concatenate([obs_embed, obs_wrist], dim=0).unsqueeze(0)

                # obs_state = state.float()
                # obs_state = agent.encoder_state.forward(obs_state)
                # cur_state = obs_state.reshape(n_step, -1)

                if step < WINDOW_SIZE:
                    action = np.zeros(7)
                    obs_total_list.append(obs_total)
                else:
                    obs_total_list = obs_total_list[1:]
                    obs_total_list.append(obs_total)

                    cur_obs_total = torch.concatenate(obs_total_list, dim=0)

                    obs_enc = agent.encoder(cur_obs_total).unsqueeze(0)

                    obs_enc = agent.encoder(cur_obs_total).flatten(start_dim=-2).unsqueeze(0)
                    goal_enc = agent.encoder(goal_obs).flatten(start_dim=-2).repeat(WINDOW_SIZE, 1).unsqueeze(0)

                    action, _, _ = agent.policy(obs_enc, goal_enc, None)

                    action = action[0, -1, 0, :].detach().cpu().numpy()

                time_step = eval_env.step(action)
                step += 1

        episode += 1

    print(f'Task:{task_name} Evaluation Time:{time.time()-eval_start_time}s Success Rate:{success/NUM_EVAL_EPI*100}%', flush=True)
    
    ### Save the evaluated success rate
    try:
        with open(RESULT_FILE_NAME, 'rb') as f:
            performance = pickle.load(f)
    except:
        performance = {}
    with open(RESULT_FILE_NAME, 'wb') as f:
        performance[i] = [task_name, success/NUM_EVAL_EPI*100]
        pickle.dump(performance, f)
    eval_env.close()
print(f'=====================End Evaluation=====================')

    