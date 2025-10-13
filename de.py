import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group, gather
import torchvision.transforms as T
from utils.quantizer import VectorQuantizer
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import random
import torch.backends.cudnn as cudnn

from utils.resnet18_encoder import ResnetEncoder
from utils.policy_head import GMMHead
from utils.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug, DataAugGroup
import utils.misc as utils
from tqdm import tqdm
 
from replay_buffer import make_replay_loader_dist
from libero.libero import benchmark
from pathlib import Path
import datetime as dt

from models.encoder.resnet import *
from models.hl_policy import *
from models.decoder import *

from models.vq_behavior_transformer.gpt import GPT, GPTConfig

from utils.libero_dataset_core import get_train_val_sliced
from utils.libero_dataset import LiberoGoalDataset
from torch.utils.data import DataLoader


#################### ARGUMENTS #####################
DEVICE = 'cuda:1'
ENCODER_PATH = "pretrained_model/encoder_6.pt"
SNAPSHOT_PATH = "pretrained_model/snapshot_6.pt"

TRAIN = True
SEED = 0

STAGE = 5

OBS_SIZE = 3
FUTURE_SIZE = 4
WINDOW_SIZE = OBS_SIZE + FUTURE_SIZE

BATCH_SIZE = 32
PRETRAIN_EPOCH = 30
FINETUNE_EPOCH = 20


now = dt.datetime.now().strftime("%m-%d_%H:%M")
s1_pt_name = "/data/libero/exp_results/de_1.pt"
s2_pt_name = "/data/libero/exp_results/de_2.pt"
#################### ARGUMENTS #####################

class MYMODEL(nn.Module):
    def __init__(self,
                 time_step=WINDOW_SIZE,
                 feature_dim=512, 
                 hidden_dim=512, 
                 output_dim=64,
                 task_embed_dim=768,
                 writer=None,
                 device=DEVICE
                 ):
        super(MYMODEL, self).__init__()
        self.device = device

        self.time_step = FUTURE_SIZE
        self.writer = writer

        self.encoder = torch.load(ENCODER_PATH).to(device)
        self.IDM  = torch.load(SNAPSHOT_PATH).to(device)

        self.hl_policy = nn.Sequential(
            GPT(
                GPTConfig(
                    block_size=OBS_SIZE,
                    n_layer=6,
                    n_head=8,
                    n_embd=feature_dim,
                    output_dim=feature_dim,
                    dropout=0.0,
                    input_dim=512 * 2 * 2,
                )
            ),
            nn.Flatten(),
            nn.Linear(feature_dim * OBS_SIZE, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 64)
        )

        self.decoder = DECODER_L(feature_dim=feature_dim*2, goal_dim=64, hidden_dim=hidden_dim).to(device)
        
    def pretrain(self, data_loader, epoch, make_log = False, save_file_name = None):
        
        optim = torch.optim.AdamW(self.hl_policy.parameters(), lr=1e-4)
        if make_log:
            writer = SummaryWriter()
        for e in tqdm(range(epoch)):
            epoch_pbar = tqdm(data_loader)
            for data in epoch_pbar:
                obs, gt_act, final_goal = data
                obs = obs.to(DEVICE)

                next_obs = obs[1:]
                obs = obs[:-1]
                gt_act = gt_act.to(DEVICE)[:-1]
                final_goal = final_goal.to(DEVICE)

                with torch.no_grad():
                    obs_enc = self.encoder(obs)  # (window_size, 2, 512)
                    goal_enc = self.encoder(final_goal)  # (window_size, 2, 512)
                    target = self.IDM(obs_enc[OBS_SIZE-1:OBS_SIZE+4].unsqueeze(0))
                    target = target[:, -1:].flatten(start_dim=1)

                goal_gpt = goal_enc[0].flatten(start_dim=0).repeat(OBS_SIZE, 1)
                
                gpt_input = torch.concat([obs_enc[:OBS_SIZE].flatten(start_dim=1), goal_gpt], dim=-1).unsqueeze(0)
                action = self.hl_policy.forward(gpt_input)

                action_loss = F.mse_loss(action, target)

                loss = action_loss
                # pred_loss = F.mse_loss(pred, obs_enc[-1].flatten(start_dim=0))
                # loss = action_loss + pred_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                # epoch_pbar.set_description("action loss : {0:.6f}   pred loss : {1:.6f}".format(action_loss.item(), pred_loss.item()))
                epoch_pbar.set_description("action loss : {0:.6f}".format(action_loss.item()))

                if not TRAIN:
                    return
                
            if save_file_name is not None and e % 10 == 0:
                torch.save(self.state_dict(), save_file_name)
                print("\nSAVE AT ", save_file_name)



    def update(self, data_loader, epoch, make_log = False, save_file_name = None):

        optim = torch.optim.AdamW(self.decoder.parameters(), lr=5e-5)

        if make_log:
            writer = SummaryWriter()

        step = 0
        for e in tqdm(range(epoch)):
            goal = None
            policy_obs = None
            policy_target = None
            epoch_pbar = tqdm(data_loader)
            for data in epoch_pbar:
                obs, gt_act, final_goal = data

                obs = obs.to(DEVICE)

                
                next_obs = obs[1:]
                obs = obs[:-1]
                gt_act = gt_act.to(DEVICE)[:-1]
                final_goal = final_goal.to(DEVICE)

                with torch.no_grad():
                    obs_enc = self.encoder(obs)
                    goal_enc = self.encoder(final_goal)
                    goal_gpt = goal_enc[0].flatten(start_dim=0).repeat(OBS_SIZE, 1)
                    gpt_input = torch.concat([obs_enc[:OBS_SIZE].flatten(start_dim=1), goal_gpt], dim=-1).unsqueeze(0)
                    goal = self.hl_policy.forward(gpt_input).flatten(start_dim=0)

                joint_loss_list = []
                gripper_loss_list = []

                for i, cur_obs in enumerate(obs_enc):

                    joint_act, gripper_act = self.decoder(cur_obs.flatten(start_dim=0), goal)

                    # print(gt_act[i, -1:])
                    joint_loss = F.mse_loss(joint_act, gt_act[i, :-1]) 
                    gripper_loss = F.binary_cross_entropy(gripper_act, (gt_act[i, -1:] + 1.) // 2) 
                    loss = joint_loss + gripper_loss

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    joint_loss_list.append(joint_loss.item())
                    gripper_loss_list.append(gripper_loss.item())
                
                epoch_pbar.set_description("joint loss : {0:.6f}   gripper loss : {1:.6f}".format(np.mean(joint_loss_list), np.mean(gripper_loss_list)))
                if make_log:
                    step += 1
                    writer.add_scalar("joint loss", np.mean(joint_loss_list), step)
                    writer.add_scalar("gripper loss", np.mean(gripper_loss_list), step)

                if not TRAIN:
                    print("Done")
                    return

            if save_file_name is not None and e % 10 == 0:
                torch.save(self.state_dict(), save_file_name)
                print("\nSAVE AT ", save_file_name)


if __name__ == "__main__":

    # Fix Random Seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)


    DATA_STORAGE_DIR = "/home/dls0901/data_prise/prise"
    TASK_DATA_DIR_SUFFIX = "_framestack1"


    if TRAIN:
            
        # Model, optimizer set
        m = MYMODEL().to(device=DEVICE)

        dataset = LiberoGoalDataset()
        data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        
        kwargs = {
            "train_fraction": 0.999,
            "random_seed": SEED,
            "window_size": WINDOW_SIZE+1,
            "future_conditional": False,
            "min_future_sep": 0,
            "future_seq_len": 0,
            "num_extra_predicted_actions": 0,
        }
        train_loader, test_loader = get_train_val_sliced(dataset, **kwargs)

        m.pretrain(train_loader, PRETRAIN_EPOCH, False, save_file_name=s1_pt_name)
        torch.save(m.state_dict(), s1_pt_name)

        del dataset
        del data_loader
        del train_loader
        del test_loader

        m.load_state_dict(torch.load(s1_pt_name))
        
        dataset = LiberoGoalDataset(subset_fraction=5)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        
        kwargs = {
            "train_fraction": 0.999,
            "random_seed": SEED,
            "window_size": WINDOW_SIZE,
            "future_conditional": False,
            "min_future_sep": 0,
            "future_seq_len": 0,
            "num_extra_predicted_actions": 0,
        }
        train_loader, test_loader = get_train_val_sliced(dataset, **kwargs)

        m.update(train_loader, FINETUNE_EPOCH, make_log=True, save_file_name=s2_pt_name)
        torch.save(m.state_dict(), s2_pt_name)


    else:
            
        # Model, optimizer set
        m = MYMODEL().to(device=DEVICE)

        dataset = LiberoGoalDataset(subset_fraction=5)
        # dataset = LiberoGoalDataset()
        data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        
        kwargs = {
            "train_fraction": 0.999,
            "random_seed": SEED,
            "window_size": WINDOW_SIZE,
            "future_conditional": False,
            "min_future_sep": 0,
            "future_seq_len": 0,
            "num_extra_predicted_actions": 0,
        }
        train_loader, test_loader = get_train_val_sliced(dataset, **kwargs)

        m.pretrain(train_loader, 1, False, None)
        m.update(train_loader, 1, False, None)
