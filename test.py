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


for i in [0, 6, 7, 16, 42]:
    SNAPSHOT_PATH = f"pretrained_model/snapshot_{i}.pt"
    IDM_PATH = f"pretrained_model/snapshot_{i}.pt"

    with Path(SNAPSHOT_PATH).open("rb") as f:
        idm = torch.load(f)
    print(idm)
        
    # with Path(IDM_PATH).open("wb") as f:
    #     torch.save(idm, f)
