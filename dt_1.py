import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf

import wandb
import pickle
from utils.libero_dataset_core import TrajectoryEmbeddingDataset, split_traj_datasets
from utils.vqbet_repro import TrajectorySlicerDataset
from utils.libero_dataset import LiberoGoalDataset

from models.encoder.resnet import *
from models.hl_policy import *
from models.decoder import *

from models.vq_behavior_transformer.gpt import GPT, GPTConfig
from models.vq_behavior_transformer.vqvae_decoder import vqvae_decoder

#################### ARGUMENTS #####################
DEVICE = 'cuda:2'
ENCODER_PATH = "pretrained_model/encoder_6.pt"
SNAPSHOT_PATH = "pretrained_model/snapshot_6.pt"

TRAIN = True
SEED = 0

STAGE = 5

OBS_SIZE = 4
FUTURE_SIZE = 4
WINDOW_SIZE = OBS_SIZE + FUTURE_SIZE

SUBSET_FRACTION_PRETRAIN = 1
SUBSET_FRACTION = 5

BATCH_SIZE = 32
PRETRAIN_EPOCH = 40
FINETUNE_EPOCH = 20


s1_pt_name = "/data/libero/exp_results/de1_1.pt"
s2_pt_name = "/data/libero/exp_results/de1_2.pt"
#################### ARGUMENTS #####################

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def main():
    seed_everything(SEED)

    encoder = torch.load(ENCODER_PATH).to(DEVICE).eval()

    dataset = LiberoGoalDataset(subset_fraction=SUBSET_FRACTION_PRETRAIN)
    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=0.99,
        random_seed=SEED,
    )
    use_libero_goal = True
    train_data = TrajectoryEmbeddingDataset(
        encoder, train_data, device=DEVICE, embed_goal=use_libero_goal
    )
    # test_data = TrajectoryEmbeddingDataset(
    #     encoder, test_data, device=DEVICE, embed_goal=use_libero_goal
    # )
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
    # test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False
    # )

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    IDM = torch.load(SNAPSHOT_PATH).to(DEVICE).eval()
    
    hl_policy = nn.Sequential(
        GPT(
            GPTConfig(
                block_size=4,
                n_layer=6,
                n_head=8,
                n_embd=512,
                output_dim=64,
                dropout=0.0,
                input_dim=512 * 2 * 2,
            )
        )
    ).to(DEVICE)

    decoder = vqvae_decoder(
        gpt_output_dim=256,
        vqvae_latent_dim=512,
        vqvae_n_embed=16,
        vqvae_groups=2,
        vqvae_fit_steps=882,
        vqvae_iters=300,
        obs_window_size=OBS_SIZE,
        act_window_size=FUTURE_SIZE
    ).to(DEVICE)

    hl_policy_optimizer = torch.optim.AdamW(hl_policy.parameters(), lr=5e-5)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5)

    with torch.no_grad():
        # calculate goal embeddings for each task
        goals_cache = []
        for i in range(10):
            idx = i * (50 // SUBSET_FRACTION_PRETRAIN)
            last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
            last_obs = last_obs.to(DEVICE)
            embd = encoder(last_obs)[0]  # V E
            embd = einops.rearrange(embd, "V E -> (V E)")
            goals_cache.append(embd)


    for epoch in tqdm.trange(PRETRAIN_EPOCH):
        hl_policy.train()
        for data in tqdm.tqdm(train_loader):
            hl_policy_optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            with torch.no_grad():
                target = IDM(obs[:, -FUTURE_SIZE:])
                target = target.flatten(start_dim=-2)

            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            gpt_input = torch.concat([goal[:, :OBS_SIZE], obs[:, :OBS_SIZE]], dim=-1)
            # print(gpt_input.shape)
            action = hl_policy.forward(gpt_input)

            loss = F.mse_loss(action, target)

            loss.backward()
            hl_policy_optimizer.step()
        torch.save(hl_policy, s1_pt_name)

    dataset = LiberoGoalDataset(subset_fraction=SUBSET_FRACTION)
    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=0.99,
        random_seed=SEED,
    )
    use_libero_goal = True
    train_data = TrajectoryEmbeddingDataset(
        encoder, train_data, device=DEVICE, embed_goal=use_libero_goal
    )
    # test_data = TrajectoryEmbeddingDataset(
    #     encoder, test_data, device=DEVICE, embed_goal=use_libero_goal
    # )
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
    # test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False
    # )

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


    for epoch in tqdm.trange(FINETUNE_EPOCH):
        decoder.train()
        for data in tqdm.tqdm(train_loader):
            decoder_optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            with torch.no_grad():
                gpt_input = torch.concat([goal[:, :OBS_SIZE], obs[:, :OBS_SIZE]], dim=-1)
                # print(gpt_input.shape)
                subgoal = hl_policy.forward(gpt_input).flatten(start_dim=-2).unsqueeze(-2)

            action, loss, loss_dict = decoder(subgoal, action_seq=act[:, -(FUTURE_SIZE):])

            loss.backward()
            decoder_optimizer.step()
        torch.save(decoder, s2_pt_name)

if __name__ == "__main__":
    main()
