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
from models.vq_behavior_transformer.vqvae_decoder import vqvae_decoder

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
s1_pt_name = "/data/libero/exp_results/deO.pt"
#################### ARGUMENTS #####################

from de_o import MYMODEL

m = MYMODEL().to(DEVICE)
torch.save(m, s1_pt_name)