import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import wandb
import pickle

from utils.libero_gym import LiberoEnv

from utils.libero_dataset_core import TrajectoryEmbeddingDataset, split_traj_datasets
from utils.vqbet_repro import TrajectorySlicerDataset
from utils.libero_dataset import LiberoGoalDataset

from models.vq_behavior_transformer.gpt import GPT, GPTConfig
from models.vq_behavior_transformer.bet_dynamo import BehaviorTransformer


if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

#################### ARGUMENTS #####################
DEVICE = 'cuda:1'
ENCODER_PATH = "pretrained_model/encoder_6.pt"
SNAPSHOT_PATH = "pretrained_model/snapshot_6.pt"

TRAIN = True
SEED = 62

STAGE = 5

OBS_SIZE = 10
FUTURE_SIZE = 0
WINDOW_SIZE = OBS_SIZE + FUTURE_SIZE

ACTION_WINDOW_SIZE = 1

BATCH_SIZE = 32
PRETRAIN_EPOCH = 50

SUBSET_FRACTION = 1

s1_pt_name = "/data/libero/exp_results/dynamo_origianl_ft_subset_1.pt"
#################### ARGUMENTS #####################

def main():
    seed_everything(SEED)

    encoder = torch.load(ENCODER_PATH).to(DEVICE).eval()

    dataset = LiberoGoalDataset(subset_fraction=SUBSET_FRACTION)

    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=0.95,
        random_seed=SEED,
    )
    use_libero_goal = True
    train_data = TrajectoryEmbeddingDataset(
        encoder, train_data, device=DEVICE, embed_goal=use_libero_goal
    )
    test_data = TrajectoryEmbeddingDataset(
        encoder, test_data, device=DEVICE, embed_goal=use_libero_goal
    )
    traj_slicer_kwargs = {
        "window": WINDOW_SIZE,
        "action_window": 1,
        "vqbet_get_future_action_chunk": True,
        "future_conditional": True,
        "min_future_sep": WINDOW_SIZE,
        "future_seq_len": 1,
        "use_libero_goal": use_libero_goal,
    }
    train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
    test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False
    )
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    cbet_model = BehaviorTransformer(
            obs_dim=512,
            act_dim=7,
            goal_dim=512,
            views=2,
            vqvae_batch_size=2048,
            vqvae_latent_dim=512,
            vqvae_n_embed=16,
            vqvae_groups=2,
            vqvae_fit_steps=940,
            vqvae_iters=600,
            n_layer=6,
            n_head=6,
            n_embd=120,
            act_scale=1,
            obs_window_size=10,
            act_window_size=1,
            offset_loss_multiplier=100
        ).to(DEVICE)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=2e-4,
        learning_rate=5.5e-5,
        betas=[0.9, 0.999]
    )

    env = LiberoEnv()

    with torch.no_grad():
        # calculate goal embeddings for each task
        goals_cache = []
        for i in range(10):
            idx = i * (50 // SUBSET_FRACTION)
            last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
            last_obs = last_obs.to(DEVICE)
            embd = encoder(last_obs)[0]  # V E
            embd = einops.rearrange(embd, "V E -> (V E)")
            goals_cache.append(embd)

    def goal_fn(goal_idx):
        return goals_cache[goal_idx]


    @torch.no_grad()
    def eval_on_env(
        num_evals=10,
        num_eval_per_goal=1,
        epoch=None
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
            eval_pbar = tqdm.tqdm(range(num_eval_per_goal))
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
                    eval_pbar.set_description("total_reward : {0}".format(avg_reward))
                avg_reward += total_reward
                completion_id_list.append(info["all_completions_ids"])
        print("eval_on_env : {}".format(avg_reward / (num_evals * num_eval_per_goal)))
        print()
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    metrics_history = []
    reward_history = []
    for epoch in tqdm.trange(PRETRAIN_EPOCH):
        cbet_model.eval()
        if epoch % 5 == 0 and epoch >= 5:
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                epoch=epoch,
                num_eval_per_goal=20,
            )
            reward_history.append(avg_reward)

        if epoch % 10 == 0 and epoch > 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(DEVICE) for x in data)
                    assert obs.ndim == 4, "expect N T V E here"
                    obs = einops.rearrange(obs, "N T V E -> N T (V E)")
                    goal = einops.rearrange(goal, "N T V E -> N T (V E)")
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}")

        cbet_model.train()
        
        train_pbar = tqdm.tqdm(train_loader)
        for data in train_pbar:
            optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            loss.backward()
            optimizer.step()
            train_pbar.set_description("EPOCH {0}, loss {1:.6f}".format(epoch, loss.item()))
        torch.save(cbet_model, s1_pt_name)

    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        num_evals=10,
        num_eval_per_goal=20,
    )
    reward_history.append(avg_reward)

    final_eval_on_env = max(reward_history)
    print("final_eval_on_env : {}".format(final_eval_on_env))
    return final_eval_on_env


if __name__ == "__main__":
    main()
