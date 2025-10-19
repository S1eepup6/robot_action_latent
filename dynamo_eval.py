import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from omegaconf import OmegaConf

import pickle
from utils.libero_dataset_core import TrajectoryEmbeddingDataset, split_traj_datasets
from utils.libero_gym import LiberoEnv

from utils.libero_dataset_core import get_train_val_sliced
from utils.libero_dataset import LiberoGoalDataset
from torch.utils.data import DataLoader

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


torch.backends.cudnn.benchmark = True
SEED = 0

# Fix Random Seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(SEED)

#################### ARGUMENTS #####################
DEVICE = 'cuda:0'
NUM_EVAL_EPI = 20
EVAL_MAX_STEP = 600

WINDOW_SIZE = 10
ACTION_WINDOW_SIZE = 1

RESULT_FILE_NAME = "performance_deo.pkl"
ENCODER_PATH = "pretrained_model/encoder_6.pt"
AGENT_PATH = "/data/libero/exp_results/deo.pt"
#################### ARGUMENTS #####################

def main():
    
    cbet_model = torch.load(AGENT_PATH).to(DEVICE)
    encoder = torch.load(ENCODER_PATH).to(DEVICE)

    dataset = LiberoGoalDataset()
    with torch.no_grad():
        # calculate goal embeddings for each task
        goals_cache = []
        for i in range(10):
            idx = i * 50
            last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
            last_obs = last_obs.to(DEVICE)
            embd = encoder(last_obs)[0]  # V E
            embd = einops.rearrange(embd, "V E -> (V E)")
            goals_cache.append(embd)

    def goal_fn(goal_idx):
        return goals_cache[goal_idx]

    env = LiberoEnv()

    @torch.no_grad()
    def eval_on_env(
        num_evals=10,
        num_eval_per_goal=1,
    ):
        def embed(enc, obs):
            obs = (
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )  # 1 V C H W
            result = enc(obs)
            result = einops.rearrange(result, "1 V E -> (V E)")
            return result

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        env.seed(SEED)
        for goal_idx in range(num_evals):
            eval_pbar = tqdm(range(num_eval_per_goal))
            for i in eval_pbar:
                obs_stack = deque(maxlen=WINDOW_SIZE)
                this_obs = env.reset(goal_idx=goal_idx)  # V C H W
                assert (
                    this_obs.min() >= 0 and this_obs.max() <= 1
                ), "expect 0-1 range observation"
                this_obs_enc = embed(encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                goal = goal_fn(goal_idx)  # V C H W
                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().to(DEVICE)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=DEVICE)
                    # goal = embed(encoder, goal)
                    goal = goal.unsqueeze(0).repeat(WINDOW_SIZE, 1)
                    action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)
                    action = action[0]  # remove batch dim; always 1
                    if ACTION_WINDOW_SIZE > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > ACTION_WINDOW_SIZE:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    step += 1
                    total_reward += reward
                    goal = goal_fn(goal_idx)
                    eval_pbar.set_description("total_reward : {0}".format(total_reward))
                avg_reward += total_reward
                completion_id_list.append(info["all_completions_ids"])
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        num_evals=10,
        num_eval_per_goal=20
    )
    print(avg_reward)


if __name__ == "__main__":
    main()
