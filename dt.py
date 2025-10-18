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
from utils.libero_dataset_core import TrajectoryEmbeddingDataset, split_traj_datasets
from utils.vqbet_repro import TrajectorySlicerDataset
from utils.libero_dataset import LiberoGoalDataset

from models.vq_behavior_transformer.gpt import GPT, GPTConfig
from models.vq_behavior_transformer.bet import BehaviorTransformer

#################### ARGUMENTS #####################
DEVICE = 'cuda:0'
ENCODER_PATH = "pretrained_model/encoder_6.pt"
SNAPSHOT_PATH = "pretrained_model/snapshot_6.pt"

TRAIN = True
SEED = 0

STAGE = 5

OBS_SIZE = 10
FUTURE_SIZE = 0
WINDOW_SIZE = OBS_SIZE + FUTURE_SIZE

BATCH_SIZE = 32
PRETRAIN_EPOCH = 30
FINETUNE_EPOCH = 40

SUBSET_FRACTION = 1

s1_pt_name = "/data/libero/exp_results/deo.pt"
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
            vqvae_fit_steps=884,
            vqvae_iters=300,
            n_layer=6,
            n_head=6,
            n_embd=120,
            act_scale=1,
            obs_window_size=10,
            act_window_size=1,
            offset_loss_multiplier=100
        ).to(DEVICE)
    optimizer = torch.optim.AdamW(cbet_model.parameters(), lr=5e-5)

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


    for epoch in tqdm.trange(PRETRAIN_EPOCH):
        cbet_model.train()
        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            obs, act, goal = (x.to(DEVICE) for x in data)
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            loss.backward()
            optimizer.step()

        torch.save(cbet_model, s1_pt_name)

if __name__ == "__main__":
    main()
