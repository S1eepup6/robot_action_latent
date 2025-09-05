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

from models.idm import IDM_MASKED_ATTN
from models.fdm import *
from models.hl_policy import *
# from models.decoder import *
from models.seer_encoder_mm import *
from models.vit import ViT

#################### ARGUMENTS #####################
DEVICE = 'cuda:1'

TRAIN = True
SEED = 0

STAGE = 5

N_HISTORY = 8
N_FUTURE_STEP = 2


now = dt.datetime.now().strftime("%m-%d_%H:%M")
s1_pt_name = "/data/libero/exp_results/s1g_seer_mms_ori.pt".format(now, SEED)
s2_pt_prefix = "/data/libero/exp_each/s2g_seer_mms_ori"
#################### ARGUMENTS #####################

class DECODER_MM(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 multi_modality = 2,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_MM, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.joint_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.eef_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        l = self.base_module(x)
        return self.joint_actor(l), self.eef_selector(l)
    
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

        self.encoder = ViT(image_size=128,
                        patch_size=16,
                        num_classes=0,
                        dim=512,
                        depth=8,
                        heads=8,
                        mlp_dim=feature_dim).to(device)
        
        self.encoder_wrist = ViT(image_size=128,
                                patch_size=16,
                                num_classes=0,
                                dim=512,
                                depth=8,
                                heads=8,
                                mlp_dim=feature_dim).to(device)
        
        self.encoder_state = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.seer_module = SEER_ENCODER_STATE(feature_dim=feature_dim, 
                                            task_embed_dim=task_embed_dim, 
                                            total_step=time_step,
                                            future_step=future_step, 
                                            attn_depth=8, 
                                            hidden_dim=hidden_dim, 
                                            output_dim=output_dim, 
                                            device=device).to(device)
        self.predictor_agent = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.predictor_wrist = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.predictor_state = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.a_quantizer = VectorQuantizer(n_code, feature_dim, device=device)

        # self.decoder = DECODER_L(feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim).to(device)

        self.decoder = DECODER_MM(feature_dim=feature_dim, hidden_dim=hidden_dim, multi_modality=3, output_dim=hidden_dim).to(device)

        self.decoder_head = GMMHead(hidden_dim, action_dim-1, hidden_dim, loss_coef=0.01)

    def update_stage1(self, obs_history, next_obs, optim, ground_truth_action, cur_step=0):
        '''
        Observations -> idm -> a_quantizer -> q_emb
        q_emb + start_obs -> fdm
        fdm ---- observations[start+1:]
        '''

        ### (batch_size, num_step_history, 3, 128, 128)
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_state, next_task_embedding = next_obs

        obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
        obs_full_history_wrist = torch.concat((obs_wrist_history, next_obs_wrist), dim=1)
        state_full_history = torch.concat((state_history, next_state), dim=1)
        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        next_time_step = next_obs_agent.shape[1]

        # task_embedding_history = torch.concat((task_embedding_history, next_task_embedding), dim=1)

        # print(idm_obs_agent.shape)
        # print(obs_wrist_history.shape)
        # print(state_history.shape)

        obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
        obs_embed = self.encoder.forward(obs_full_history)
        obs_embed = obs_embed.reshape(batch_size, time_step+next_time_step, -1)

        obs_full_history_wrist = obs_full_history_wrist.reshape(-1, num_channel, img_size, img_size).float()
        obs_embed_wrist = self.encoder_wrist.forward(obs_full_history_wrist)
        obs_embed_wrist = obs_embed_wrist.reshape(batch_size, time_step+next_time_step, -1)
        
        state_full_history = state_full_history.float()
        obs_embed_state = self.encoder_state.forward(state_full_history)

        seer_input_agent = obs_embed[:, :-next_time_step]
        seer_input_wrist = obs_embed_wrist[:, :-next_time_step]
        seer_input_state = obs_embed_state[:, :-next_time_step]

        action_z, pred_agent, pred_wrist, pred_state = self.seer_module.forward(seer_input_agent, seer_input_wrist, seer_input_state, task_embedding_history[:, 0].unsqueeze(-2).float())
        pred_agent = self.predictor_agent(pred_agent)
        pred_wrist = self.predictor_wrist(pred_wrist)
        pred_state = self.predictor_state(pred_state)

        ans = obs_embed[:, time_step:]
        ans_wrist = obs_embed_wrist[:, time_step:]
        ans_state = obs_embed_state[:, time_step:]

        decoder_input = action_z
        joint_action, eef_action = self.decoder(decoder_input)
        joint_action = self.decoder_head(joint_action)
        eef_action = eef_action.squeeze(dim=-1)

        gt_joint = ground_truth_action[:, :-1]
        gt_eef = (ground_truth_action[:, -1] + 1.) // 2

        joint_loss = self.decoder_head.loss_fn(joint_action, gt_joint, reduction='none')
        eef_loss = F.binary_cross_entropy(eef_action, gt_eef)

        loss = torch.mean(joint_loss) + eef_loss + F.mse_loss(pred_agent, ans) + F.mse_loss(pred_wrist, ans_wrist) + F.mse_loss(pred_state, ans_state)

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()


    def update(self, next_data, optim, stage=1, cur_step=0):
        
        obs_history, action, action_seq, next_obs = next_data
        obs_history = utils.to_torch(obs_history, device=self.device)
        next_obs    = utils.to_torch(next_obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device).float()

        # print(obs_history[0].shape)
        # print(next_obs[0].shape)


        if stage == 1:
            return self.update_stage1(obs_history, next_obs, optim, action, cur_step=cur_step)
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

        pretraining_data_dirs = []
        for task_id in range(1): 
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_goal']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs.append(offline_data_dir)
        eval_env = None

        replay_loader = make_replay_loader_dist(
                        replay_dir=pretraining_data_dirs, max_traj_per_task=50, max_size=100000000,
                        batch_size=64//1, num_workers=4,
                        save_snapshot=True, nstep=N_FUTURE_STEP, nstep_history=N_HISTORY, 
                        rank=0, world_size=1, n_code=10, vocab_size=200,
                        min_frequency=5, max_token_length=20)


        replay_dataset = iter(replay_loader)


        # Tensorboard Writer
        writer = SummaryWriter()

            
        # Model, optimizer set
        m = MYMODEL(writer=writer).to(device=DEVICE)
        optim = torch.optim.Adam(m.parameters(), lr=0.001)

        # Stage 1 
        for i in tqdm(range(200010)):
            loss = m.update(next(replay_dataset), optim, stage=1, cur_step=i)
            writer.add_scalar("loss/stage1", loss, i)

            if i % 10000 == 0:
                torch.save(m.state_dict(), s1_pt_name)
        torch.save(m.state_dict(), s1_pt_name)
        
        del replay_loader
        del replay_dataset
            
        # Stage 2
        for task_id in range(10): 
            print(task_id)
            # s2_pt_name = "/data/libero/exp_each/s2_seer_task_{}.pt".format(task_id)
            s2_pt_name = s2_pt_prefix + "_task_{}.pt".format(task_id)

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_goal']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs = [offline_data_dir, ]
            
            replay_loader = make_replay_loader_dist(
                            replay_dir=pretraining_data_dirs, max_traj_per_task=50, max_size=100000000,
                            batch_size=64//1, num_workers=4,
                            save_snapshot=True, nstep=N_FUTURE_STEP, nstep_history=N_HISTORY, 
                            rank=0, world_size=1, n_code=10, vocab_size=200,
                            min_frequency=5, max_token_length=20)
            replay_dataset = iter(replay_loader)
            eval_env = None

            m.load_state_dict(torch.load(s1_pt_name))

            for i in tqdm(range(30000)):
                loss = m.update(next(replay_dataset), optim, stage=1, cur_step=i)
                writer.add_scalar("loss/stage2_{}".format(task_id), loss, i)
                if i % 10000 == 0:
                    torch.save(m.state_dict(), s2_pt_name)
            torch.save(m.state_dict(), s2_pt_name)

        writer.close()

############################### Code Test ###############################
    else:
        # writer = SummaryWriter()

        m = MYMODEL().to(device=DEVICE)
        optim = torch.optim.Adam(m.parameters(), lr=0.001)

        pretraining_data_dirs = []
        for task_id in range(1): 
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_goal']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs.append(offline_data_dir)
        eval_env = None
        replay_loader = make_replay_loader_dist(
                        replay_dir=pretraining_data_dirs, max_traj_per_task=10, max_size=10000000,
                        batch_size=64//1, num_workers=4,
                        save_snapshot=True, nstep=N_FUTURE_STEP, nstep_history=N_HISTORY, 
                        rank=0, world_size=1, n_code=10, vocab_size=200,
                        min_frequency=5, max_token_length=20)
        replay_dataset = iter(replay_loader)
        
        print(m.update(next(replay_dataset), optim, stage=1))


        # writer.close()