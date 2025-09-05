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

from model_taskrevise import *

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

DEVICE = 'cuda:2'
DOWNSTREAM_TASK_SUITE = 'libero_90'
NUM_EVAL_EPI = 20
EVAL_MAX_STEP = 600


for i in tqdm(range(45)):
    agent = MYMODEL().to(DEVICE)

    agent.train(False)
    agent.decoder_head.training = False
    agent.decoder_head.low_eval_noise = True
    
    agent.a_quantizer.to(DEVICE)
    eval_until_episode = utils.Until(NUM_EVAL_EPI)
    # num_tasks_per_gpu = (90//self.world_size) + 1

    agent_path = "/data/libero/exp_each/s3_task_revise_{}.pt".format(i)
    print(agent_path)
    agent.load_state_dict(torch.load(agent_path))
    
    ### Setup eval environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[DOWNSTREAM_TASK_SUITE]()
    task = task_suite.get_task(i)
    eval_env = libero_wrapper.make(i, DOWNSTREAM_TASK_SUITE, 
                                    seed=SEED)
    eval_env.task_name = task.name
    eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)
    task_name = task.name
    
    eval_start_time = time.time()
    eval_until_episode = utils.Until(NUM_EVAL_EPI)

    counter, episode, success = 0, 0, 0
    while eval_until_episode(episode):
        time_step = eval_env.reset()
        step, q_res = 0, None
        while step < EVAL_MAX_STEP:
            if time_step['done']:
                success += 1
                break
            with torch.no_grad():
                obs_agent = time_step.agentview
                obs_wrist = time_step.wristview
                state     = time_step.state 
                task_embedding = eval_env.task_embedding
                
                ### Encode the current timestep
                task_embedding = torch.torch.as_tensor(task_embedding, device=DEVICE)
                obs_agent = torch.torch.as_tensor(obs_agent.copy(), device=DEVICE).unsqueeze(0)
                obs_wrist = torch.torch.as_tensor(obs_wrist.copy(), device=DEVICE).unsqueeze(0)
                state     = torch.torch.as_tensor(state, device=DEVICE).unsqueeze(0)

                n_step, num_channel, img_size = obs_agent.shape[0], obs_agent.shape[1], obs_agent.shape[2]
                
                obs_embed = obs_agent.reshape(-1, num_channel, img_size, img_size).float()
                obs_embed = agent.encoder.forward(obs_embed)
                cur_obs = obs_embed.reshape(n_step, -1)

                if step % 4 == 0 or q_res is None:
                    hl_policy_input = torch.concat([cur_obs, task_embedding], dim=1).float()

                    z_policy = agent.hl_policy(hl_policy_input)   
                    q_loss, z_q, _, _, _ = agent.a_quantizer.forward(z_policy)


                # decoder_input = torch.concat([cur_obs, z_q], dim=1)
                action_step_i = step % 4
                action = agent.decoder(z_q)[:, 512*(action_step_i):512*(action_step_i+1)]
                action = agent.decoder_head(action)
                action = action.sample()

                action = action.detach().cpu().numpy()[0]

            time_step = eval_env.step(action)
            step += 1
        episode += 1

    print(f'Task:{task_name} Evaluation Time:{time.time()-eval_start_time}s Success Rate:{success/NUM_EVAL_EPI*100}%', flush=True)
    
    ### Save the evaluated success rate
    try:
        with open("performance_task_revise.pkl", 'rb') as f:
            performance = pickle.load(f)
    except:
        performance = {}
    with open("performance_task_revise.pkl", 'wb') as f:
        performance[i] = [task_name, success/NUM_EVAL_EPI*100]
        pickle.dump(performance, f)
    eval_env.close()
print(f'=====================End Evaluation=====================')

    