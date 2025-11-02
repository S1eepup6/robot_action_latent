import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf

import wandb
import pickle

from utils.libero_gym import LiberoEnv

from utils.libero_dataset_core import TrajectoryEmbeddingDataset, split_traj_datasets
from utils.vqbet_repro import TrajectorySlicerDataset
from utils.libero_dataset import LiberoGoalDataset

from models.vq_behavior_transformer.gpt import GPT, GPTConfig
from models.vq_behavior_transformer.vqvae_decoder import vqvae_decoder
from models.vq_behavior_transformer.bet import BehaviorTransformer


if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

#################### ARGUMENTS #####################
DEVICE = 'cuda:2'
ENCODER_PATH = "pretrained_model/encoder_6.pt"
SNAPSHOT_PATH = "pretrained_model/snapshot_6.pt"

TRAIN = True
SEED = 42

STAGE = 5

OBS_SIZE = 10
FUTURE_SIZE = 8
WINDOW_SIZE = OBS_SIZE + FUTURE_SIZE

ACTION_WINDOW_SIZE = 1

BATCH_SIZE = 32
if TRAIN:
    NUM_EVAL_PER_GOAL = 10
    PRETRAIN_EPOCH = 50
    FINETUNE_EPOCH = 50
    SUBSET_FRACTION_PRETRAIN = 1
    SUBSET_FRACTION_FINETUNE = 2
else: # TEST 
    NUM_EVAL_PER_GOAL = 1
    PRETRAIN_EPOCH = 1
    FINETUNE_EPOCH = 2
    SUBSET_FRACTION_PRETRAIN = 25
    SUBSET_FRACTION_FINETUNE = 25

s1_pt_name = "/data/libero/exp_results/do1_1.pt"
s2_pt_name = "/data/libero/exp_results/do1_2.pt"
#################### ARGUMENTS #####################

def main():
    seed_everything(SEED)

    encoder = torch.load(ENCODER_PATH).to(DEVICE).eval()

    dataset = LiberoGoalDataset(subset_fraction=SUBSET_FRACTION_FINETUNE)

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

    IDM = torch.load(SNAPSHOT_PATH).to(DEVICE).eval()
    
    hl_policy = GPT(
            GPTConfig(
                block_size=OBS_SIZE,
                n_layer=6,
                n_head=8,
                n_embd=512,
                output_dim=1024,
                dropout=0.0,
                input_dim=512 * 2 * 2,
            )
        ).to(DEVICE)

    decoder = BehaviorTransformer(
            obs_dim=512,
            act_dim=7,
            goal_dim=512,
            views=2,
            vqvae_batch_size=2048,
            vqvae_latent_dim=512,
            vqvae_n_embed=16,
            vqvae_groups=2,
            vqvae_fit_steps=941,
            vqvae_iters=600,
            n_layer=6,
            n_head=6,
            n_embd=120,
            act_scale=1,
            obs_window_size=OBS_SIZE,
            act_window_size=FUTURE_SIZE,
            offset_loss_multiplier=100
        ).to(DEVICE)

    hl_policy_optimizer = hl_policy.configure_optimizers(
        weight_decay=2e-4,
        learning_rate=5.5e-5,
        betas=[0.9, 0.999]
    )
    decoder_optimizer = decoder.configure_optimizers(
        weight_decay=2e-4,
        learning_rate=5.5e-5,
        betas=[0.9, 0.999]
    )
    env = LiberoEnv()

    with torch.no_grad():
        # calculate goal embeddings for each task
        goals_cache = []
        for i in range(10):
            idx = i * (50 // SUBSET_FRACTION_FINETUNE)
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
                obs_stack = deque(maxlen=OBS_SIZE)
                this_obs = env.reset(goal_idx=goal_idx)  # V C H W
                assert (
                    this_obs.min() >= 0 and this_obs.max() <= 1
                ), "expect 0-1 range observation"
                this_obs_enc = embed(encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                goal = goal_fn(goal_idx)  # V C H W

                for i in range(OBS_SIZE):
                    this_obs, reward, done, info = env.step(np.zeros(7))
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().unsqueeze(0).to(DEVICE)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=DEVICE)
                    # goal = embed(encoder, goal)
                    goal = goal.unsqueeze(0).repeat(WINDOW_SIZE, 1).unsqueeze(0)

                    if step % FUTURE_SIZE == 0:
                        gpt_input = torch.concat([goal[:, :OBS_SIZE], obs[:, :OBS_SIZE]], dim=-1)
                        subgoal = hl_policy.forward(gpt_input)[:, -4:].flatten(start_dim=-2).unsqueeze(-2)
                        subgoal = subgoal.repeat(OBS_SIZE, 1, 1).transpose(1, 0)
                        action, _, _ = decoder(obs, subgoal, action_seq=None)
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
                        curr_action = action[-1, (-FUTURE_SIZE + (step % FUTURE_SIZE)), :].cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    step += 1
                    total_reward += reward
                    goal = goal_fn(goal_idx)
                    eval_pbar.set_description("total_reward : {0}".format(avg_reward))
                avg_reward += total_reward
                completion_id_list.append(info["all_completions_ids"])
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
        hl_policy.train()
        pretrain_pbar = tqdm.tqdm(train_loader)
        for data in pretrain_pbar:
            hl_policy_optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            # with torch.no_grad():
            #     target = IDM(obs[:, -4:])
            #     target = target.flatten(start_dim=-2)

            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            gpt_input = torch.concat([goal[:, :OBS_SIZE], obs[:, :OBS_SIZE]], dim=-1)
            # print(gpt_input.shape)
            subgoal = hl_policy.forward(gpt_input)[:, -1:]

            loss = F.mse_loss(subgoal, obs[:, -1:])

            loss.backward()
            hl_policy_optimizer.step()

            pretrain_pbar.set_description("loss {0:.6f}".format(loss.item()))
        torch.save(hl_policy, s1_pt_name)

    del dataset
    del train_data
    del test_data
    del train_loader
    del test_loader

    dataset = LiberoGoalDataset(subset_fraction=SUBSET_FRACTION_FINETUNE)

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

    with torch.no_grad():
        # calculate goal embeddings for each task
        goals_cache = []
        for i in range(10):
            idx = i * (50 // SUBSET_FRACTION_FINETUNE)
            last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
            last_obs = last_obs.to(DEVICE)
            embd = encoder(last_obs)[0]  # V E
            embd = einops.rearrange(embd, "V E -> (V E)")
            goals_cache.append(embd)

    for epoch in tqdm.trange(FINETUNE_EPOCH):
        decoder.eval()
        if epoch % 5 == 0 and epoch > 0:
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                epoch=epoch,
                num_eval_per_goal=NUM_EVAL_PER_GOAL,
            )
            reward_history.append(avg_reward)

        decoder.train()
        finetune_pbar = tqdm.tqdm(train_loader)
        for data in finetune_pbar:
            decoder_optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            with torch.no_grad():
                gpt_input = torch.concat([goal[:, :OBS_SIZE], obs[:, :OBS_SIZE]], dim=-1).to(DEVICE)
                # print(gpt_input.shape)
                subgoal = hl_policy.forward(gpt_input)[:, -1:].flatten(start_dim=-2) # N 1024
                subgoal = subgoal.repeat(OBS_SIZE, 1, 1).transpose(1, 0) # N T 1024

            # print(obs.shape, subgoal.shape)
            action, loss, loss_dict = decoder(obs[:, :OBS_SIZE], subgoal, action_seq=act[:, -FUTURE_SIZE:].to(DEVICE))

            loss.backward()
            decoder_optimizer.step()
            finetune_pbar.set_description("EPOCH {0}, loss {1:.6f}".format(epoch, loss.item()))
        torch.save(decoder, s2_pt_name)

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
