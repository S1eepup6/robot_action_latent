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

from utils.libero_dataset_core import get_train_val_sliced
from utils.libero_dataset import LiberoGoalDataset
from torch.utils.data import DataLoader


#################### ARGUMENTS #####################
DEVICE = 'cuda:1'
ENCODER_PATH = "pretrained_model/encoder_42.pt"

TRAIN = True
SEED = 0

STAGE = 5

N_HISTORY = 4
N_FUTURE_STEP = N_HISTORY

BATCH_SIZE = 32
N_EPOCH = 60


now = dt.datetime.now().strftime("%m-%d_%H:%M")
s1_pt_name = "/data/libero/exp_results/de.pt"
s2_pt_prefix = "/data/libero/exp_each/sbm2m"
#################### ARGUMENTS #####################

class MYMODEL(nn.Module):
    def __init__(self,
                 time_step=5,
                 future_step=1,
                 obs_shape=[3, 128, 128],
                 n_code=4096, 
                 feature_dim=512, 
                 hidden_dim=512, 
                 output_dim=512,
                 task_embed_dim=768,
                 action_dim=7,
                 writer=None,
                 device=DEVICE
                 ):
        super(MYMODEL, self).__init__()
        self.device = device

        self.time_step = time_step
        self.future_step = future_step
        self.total_step = time_step + future_step
        self.writer = writer

        self.encoder = torch.load(ENCODER_PATH).to(device)

        self.hl_policy = HL_BASE(feature_dim=feature_dim, task_embed_dim=task_embed_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

        self.decoder = DECODER_MM(feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    def update(self, data_loader, epoch, save_file_name = None):
        '''
        cur_obs + task_emb -> hl_policy -> a_quantizer -> skill
        skill + cur_obs -> decoder -> action
        action ----- ground_truth?
        '''

        optim = torch.optim.Adam(self.parameters(), lr=0.001)

        for _ in tqdm(range(epoch)):
            goal = None
            step = 0
            for data in tqdm(data_loader):
                obs, gt_act, _ = data

                with torch.no_grad():
                    obs_enc = self.encoder(obs.to(DEVICE))
                # obs_enc.shape == (window_size, 2, 512)

                if goal is None or step % N_FUTURE_STEP == 0:
                    goal = self.hl_policy(obs_enc)

                joint_act, gripper_act = self.decoder(obs_enc, goal)
                loss = F.mse_loss(joint_act, gt_act[:, :-1]) + F.binary_cross_entropy(gripper_act, gt_act[:, -1])

                optim.zero_grad()
                loss.backward()
                optim.step()

                step += 1

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

        # dataset = LiberoGoalDataset(subset_fraction=0.2)
        dataset = LiberoGoalDataset()
        data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        
        kwargs = {
            "train_fraction": 0.95,
            "random_seed": SEED,
            "window_size": N_HISTORY,
            "future_conditional": False,
            "min_future_sep": 0,
            "future_seq_len": 0,
            "num_extra_predicted_actions": 0,
        }
        train_loader, test_loader = get_train_val_sliced(dataset, **kwargs)

        # Stage 1 
        m.update(train_loader, N_EPOCH, s1_pt_name)

        # torch.save(m.state_dict(), s1_pt_name)


    else:
        pass
