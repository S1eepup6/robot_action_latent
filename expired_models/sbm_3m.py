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
 
from replay_buffer_state import make_replay_loader_dist
from libero.libero import benchmark
from pathlib import Path

import datetime as dt
from functools import partial

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from models.idm import IDM_MASKED_ATTN
from models.fdm import *
from models.hl_policy import *
# from models.decoder import *
from models.seer_encoder_mm import *
from models.vit import ViT

from seer_models.vit_mae import MaskedAutoencoderViT
from seer_models.perceiver_resampler import PerceiverResampler
from seer_models.gpt2 import GPT2Model
from seer_models.seer_model_revise2 import SeerAgent

from seer_model_train_util import train_one_epoch_libero
from utils.seer_dataset import get_liberogoal_dataset
import clip

#################### ARGUMENTS #####################
DEVICE = 'cuda:2'

TRAIN = True
SEED = 0

STAGE = 5

N_HISTORY = 8
N_FUTURE_STEP = 2

BATCH_SIZE = 32
N_EPOCH = 40


now = dt.datetime.now().strftime("%m-%d_%H:%M")
s1_pt_name = "/data/libero/exp_results/sbm_r2m_1.pt".format(now, SEED)
s2_pt_prefix = "/data/libero/exp_each/sbm_r2m_2"
#################### ARGUMENTS #####################
    
def construct_task_data_path(root_dir, task_name, task_data_dir_suffix='framestack1'):
    return Path(root_dir) / (task_name.lower()+('' if not task_data_dir_suffix or task_data_dir_suffix == 'None' else task_data_dir_suffix))

def replay_iter(replay_loader):
    return iter(replay_loader)

class MYMODEL(nn.Module):
    def __init__(self,
                 time_step=N_HISTORY,
                 future_step=N_FUTURE_STEP,
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

        self.feature_dim = feature_dim

        # vision encoder (frozen)
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        self.encoder_state = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.seer_model = SeerAgent(
            finetune_type="libero_pretrain",
            clip_device=DEVICE,
            vit_checkpoint_path="/home/dls0901/seer/Seer/checkpoints/vit_mae/mae_pretrain_vit_base.pth",
            sequence_length=time_step,
            num_resampler_query=6,
            num_obs_token_per_image=9,
            calvin_input_image_size=224,
            patch_size=16,
            action_pred_steps=1,
            obs_pred=True,
            atten_only_obs=True,
            attn_robot_proprio_state=False,
            atten_goal=2,
            atten_goal_state=True,
            mask_l_obs_ratio=0.5,
            transformer_layers=12,
            hidden_dim=hidden_dim,
            transformer_heads=8,
            phase="finetune",
            gripper_width=True,
        )
        self.a_quantizer = VectorQuantizer(n_code, feature_dim, device=device)

        self.decoder_head = GMMHead(hidden_dim, action_dim-1, hidden_dim, loss_coef=0.01)

    def update_stage1(self, libero_dataset, num_epochs):
        num_batches = libero_dataset.dataloader.num_batches * 1
        optim = torch.optim.AdamW(self.seer_model.parameters(), lr=1e-4, weight_decay=1e-4)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=num_batches,
            num_training_steps=num_batches * num_epochs,
        )
        
        for i in range(num_epochs):
            libero_dataset.set_epoch(i)
            libero_loader = libero_dataset.dataloader
            train_one_epoch_libero(
                num_batches=num_batches,
                num_epochs=num_epochs,
                sequence_length=self.time_step,
                future_steps=self.future_step,
                atten_goal=2,
                window_size=self.time_step,
                action_pred_steps=1,
                patch_size=16,
                model=self.seer_model,
                epoch=i,
                libero_loader=libero_loader,
                optimizer=optim,
                lr_scheduler=lr_scheduler,
                device_id=self.device
            )
            torch.save(self.state_dict(), s1_pt_name)


    def update(self, libero_loader, num_epochs, stage=1):
        if stage == 1:
            return self.update_stage1(libero_loader, num_epochs)
        else:
            raise NotImplementedError


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


    DATA_STORAGE_DIR = "/home/dls0901/data_libero_goal"
    TASK_DATA_DIR_SUFFIX = "_framestack1"

    print(s1_pt_name)
    print(s2_pt_prefix + "_task_{}.pt".format(0))

    if TRAIN:
        assert STAGE in [1, 2, 5], "Stage must be 1, 2"

        m = MYMODEL().to(device=DEVICE)

        libero_dataset = get_liberogoal_dataset(
            seed=SEED, 
            batch_size=BATCH_SIZE,
            window_size=N_HISTORY,
            image_processor=m.seer_model.image_processor, 
            tokenizer=clip, 
            epoch=0,
            start_epi=0,
            end_epi=9
            )
        
        m.update(libero_dataset, N_EPOCH, stage=1)
        torch.save(m.state_dict(), s1_pt_name)
        
        del libero_dataset
            
        # # Stage 2
        # for task_id in range(10): 
        #     print(task_id)
        #     # s2_pt_name = "/data/libero/exp_each/s2_seer_task_{}.pt".format(task_id)
        #     s2_pt_name = s2_pt_prefix + "_task_{}.pt".format(task_id)

        #     libero_dataset = get_liberogoal_dataset(
        #         seed=SEED, 
        #         batch_size=BATCH_SIZE,
        #         window_size=N_HISTORY,
        #         image_processor=m.seer_model.image_processor, 
        #         tokenizer=clip, 
        #         epoch=task_id,
        #         start_epi=task_id,
        #         end_epi=9
        #         )

        #     m.load_state_dict(torch.load(s1_pt_name))
                
        #     m.update(libero_dataset, N_EPOCH // 2, stage=1)
        #     torch.save(m.state_dict(), s2_pt_name)

        # # writer.close()

############################### Code Test ###############################
    else:
        # writer = SummaryWriter()

        m = MYMODEL().to(device=DEVICE)

        libero_dataset = get_liberogoal_dataset(
            seed=SEED, 
            batch_size=BATCH_SIZE,
            window_size=N_HISTORY,
            image_processor=m.seer_model.image_processor, 
            tokenizer=clip, 
            epoch=0,
            start_epi=1,
            end_epi=1
            )
        libero_dataset.set_epoch(1)
        libero_loader = libero_dataset.dataloader
        
        m.update(libero_dataset, 1, stage=1)


        # writer.close()