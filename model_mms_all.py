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

from models.idm_mm import IDM_SEER_NORM, IDM_SEER_STATE
from models.fdm_mm import FDM_SEER_NORM, FDM_SEER_STATE
from models.hl_policy import *
from models.decoder import *

#################### ARGUMENTS #####################
DEVICE = 'cuda:0'

TRAIN = True
SEED = 0

STAGE = 5

N_HISTORY = 8
N_FUTURE_STEP = 2


now = dt.datetime.now().strftime("%m-%d_%H:%M")
s1_pt_name = "/data/libero/exp_results/s1g_mms_all.pt".format(now, SEED)
s2_pt_name = "/data/libero/exp_results/s2g_mms_all.pt".format(now, SEED)
s3_pt_prefix = "/data/libero/exp_each/s3g_mms_all"
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

        self.feature_dim = feature_dim

        self.time_step = time_step
        self.future_step = future_step
        self.total_step = time_step + future_step

        self.encoder = ResnetEncoder(input_shape=obs_shape, output_size=feature_dim).to(device)
        self.wrist_encoder = ResnetEncoder(input_shape=obs_shape, output_size=feature_dim).to(device)
        self.state_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )


        self.idm = IDM_SEER_STATE(feature_dim=feature_dim, 
                                  task_embed_dim=task_embed_dim, 
                                  total_step=self.total_step, 
                                  attn_depth=8,
                                  hidden_dim=hidden_dim, 
                                  output_dim=output_dim, 
                                  device=device).to(device)
        
        self.fdm = FDM_SEER_STATE(feature_dim=feature_dim, 
                                  task_embed_dim=task_embed_dim, 
                                  total_step=self.time_step, 
                                  multi_modality=3, 
                                  future_step=self.future_step,
                                  attn_depth=8,
                                  hidden_dim=hidden_dim, 
                                  output_dim=output_dim,
                                  device=device).to(device)
        
        self.a_quantizer = VectorQuantizer(n_code, feature_dim, device=device)


        self.hl_policy = HL_PRED_MMS(feature_dim=feature_dim, task_embed_dim=task_embed_dim, hidden_dim=[hidden_dim*2, hidden_dim*2, hidden_dim, hidden_dim], output_dim=output_dim).to(device)

        # self.decoder = DECODER_L_MM(feature_dim=feature_dim, hidden_dim=hidden_dim, multi_modality=3, output_dim=output_dim).to(device)
        self.decoder_head = GMMHead(hidden_dim, action_dim-1, hidden_dim, loss_coef=0.01)

        self.decoder = DECODER_MM(feature_dim=feature_dim, hidden_dim=hidden_dim, multi_modality=3, output_dim=hidden_dim).to(device)

    def update_stage1(self, obs_history, next_obs, optim, cur_step=0):
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
        obs_full_history_state = torch.concat((state_history, next_state), dim=1)
        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        next_time_step = next_obs_agent.shape[1]

        # task_embedding_history = torch.concat((task_embedding_history, next_task_embedding), dim=1)

        # print(idm_obs_agent.shape)
        # print(obs_wrist_history.shape)
        # print(task_embedding_history.shape)

        obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
        obs_embed = self.encoder.forward(obs_full_history)
        obs_embed = obs_embed.reshape(batch_size, time_step+next_time_step, -1)
        
        obs_full_history_wrist = obs_full_history_wrist.reshape(-1, num_channel, img_size, img_size).float()
        obs_embed_wrist = self.wrist_encoder.forward(obs_full_history_wrist)
        obs_embed_wrist = obs_embed_wrist.reshape(batch_size, time_step+next_time_step, -1)
        
        obs_embed_state = self.state_encoder.forward(obs_full_history_state.float())
        obs_embed_state = obs_embed_state.reshape(batch_size, time_step+next_time_step, -1)

        fdm_embed = obs_embed[:, :-next_time_step]
        ans = obs_embed[:, time_step:]
        fdm_embed_wrist = obs_embed_wrist[:, :-next_time_step]
        ans_wrist = obs_embed_wrist[:, time_step:]
        fdm_embed_state = obs_embed_state[:, :-next_time_step]
        ans_state = obs_embed_state[:, time_step:]

        z = self.idm.forward(obs_embed, obs_embed_wrist, obs_embed_state, task_embedding_history[:, 0].float().unsqueeze(-2))
        q_loss, z_q, _, _, _ = self.a_quantizer.forward(z)

        fdm_agent, fdm_wrist, fdm_state = self.fdm.forward(fdm_embed, fdm_embed_wrist, fdm_embed_state, task_embedding_history[:, 0].float().unsqueeze(-2), z_q.unsqueeze(-2))
        fdm_agent = fdm_agent.reshape(batch_size, next_time_step, -1)
        fdm_wrist = fdm_wrist.reshape(batch_size, next_time_step, -1)
        fdm_state = fdm_state.reshape(batch_size, next_time_step, -1)

        loss = F.mse_loss(fdm_agent, ans) + F.mse_loss(fdm_wrist, ans_wrist) + F.mse_loss(fdm_state, ans_state) + q_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()


    def update_stage2(self, obs_history, next_obs, optim, cur_step=0):
        '''
        Observations -> idm(frozen) -> a_quantizer(frozen) -> skill
        cur_obs + task_emb -> hl_policy -> a_quantizer(frozen) ---- skill
        '''
        
        ### (batch_size, num_step_history, 3, 128, 128)
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_state, next_task_embedding = next_obs

        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        
        with torch.no_grad():
            obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
            obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed = self.encoder.forward(obs_full_history)
            obs_embed = obs_embed.reshape(batch_size, time_step+self.future_step, -1)
                
            obs_full_history_wrist = torch.concat((obs_wrist_history, next_obs_wrist), dim=1)
            obs_full_history_wrist = obs_full_history_wrist.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed_wrist = self.wrist_encoder.forward(obs_full_history_wrist)
            obs_embed_wrist = obs_embed_wrist.reshape(batch_size, time_step+self.future_step, -1)
            
            obs_full_history_state = torch.concat((state_history, next_state), dim=1)
            obs_full_history_state = obs_full_history_state.float()
            obs_embed_state = self.state_encoder.forward(obs_full_history_state)
            obs_embed_state = obs_embed_state.reshape(batch_size, time_step+self.future_step, -1)

            z = self.idm.forward(obs_embed, obs_embed_wrist, obs_embed_state, task_embedding_history[:, 0].unsqueeze(-2))


        cur_obs = obs_embed[:, 0].flatten(start_dim=1)
        target_obs = obs_embed[:, -1].flatten(start_dim=1)

        cur_obs_wrist = obs_embed_wrist[:, 0].flatten(start_dim=1)
        target_obs_wrist = obs_embed_wrist[:, -1].flatten(start_dim=1)

        cur_obs_state = obs_embed_state[:, 0].flatten(start_dim=1)
        target_obs_state = obs_embed_state[:, -1].flatten(start_dim=1)

        ans = torch.concat([target_obs, target_obs_wrist, target_obs_state], dim=-1)

        cur_task = task_embedding_history[:, 0]
        hl_policy_input = torch.concat([cur_obs, cur_obs_wrist, cur_obs_state, cur_task], dim=1).float()

        z_policy, pred_embed = self.hl_policy.forward_with_pred(hl_policy_input)   

        loss_policy = F.mse_loss(z_policy, z)
        loss_pred = F.mse_loss(pred_embed, ans)

        loss = loss_policy + loss_pred

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()

    def update_stage3(self, obs_history, next_obs, optim, ground_truth_action, cur_step=0):
        '''
        cur_obs + task_emb -> hl_policy -> a_quantizer -> skill
        skill + cur_obs -> decoder -> action
        action ----- ground_truth?
        '''

        ### (batch_size, num_step_history, 3, 128, 128)
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_state, next_task_embedding = next_obs

        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        

        with torch.no_grad():
            obs_steps = []
            next_steps = []

            obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
            obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed = self.encoder.forward(obs_full_history)
            obs_embed = obs_embed.reshape(batch_size, time_step+self.future_step, -1)

            
            obs_full_history_wrist = torch.concat((obs_wrist_history, next_obs_wrist), dim=1)
            obs_full_history_wrist = obs_full_history_wrist.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed_wrist = self.wrist_encoder.forward(obs_full_history_wrist)
            obs_embed_wrist = obs_embed_wrist.reshape(batch_size, time_step+self.future_step, -1)
            
            obs_full_history_state = torch.concat((state_history, next_state), dim=1)
            obs_full_history_state = obs_full_history_state.float()
            obs_embed_state = self.state_encoder.forward(obs_full_history_state)
            obs_embed_state = obs_embed_wrist.reshape(batch_size, time_step+self.future_step, -1)

            first_obs = obs_embed[:, 0].flatten(start_dim=1)
            first_obs_wrist = obs_embed_wrist[:, 0].flatten(start_dim=1)
            first_obs_state = obs_embed_state[:, 0]
            cur_task = task_embedding_history[:, 0]
            
            for i in range(self.time_step):
                obs_steps.append([obs_embed[:, i].flatten(start_dim=1), obs_embed_wrist[:, i].flatten(start_dim=1), obs_embed_state[:, i].flatten(start_dim=1)])
                next_steps.append([obs_embed[:, i+1].flatten(start_dim=1), obs_embed_wrist[:, i+1].flatten(start_dim=1), obs_embed_state[:, i+1].flatten(start_dim=1)])

            hl_policy_input = torch.concat([first_obs, first_obs_wrist, first_obs_state, cur_task], dim=1).float()


        for cur_obs, next_obs in zip(obs_steps, next_steps):
            z_policy = self.hl_policy.forward(hl_policy_input)
            q_loss, z_q, _, _, _ = self.a_quantizer.forward(z_policy)

            decoder_input = torch.concat([*cur_obs, z_q], dim=1)
            joint_action, eef_action = self.decoder(decoder_input)
            joint_action = self.decoder_head(joint_action)

            gt_joint = ground_truth_action[:, :-1]
            gt_eef = (ground_truth_action[:, -1] + 1.) // 2

            eef_loss = F.cross_entropy(eef_action, gt_eef)

            joint_loss = self.decoder_head.loss_fn(joint_action, gt_joint, reduction='none')
            action_loss = torch.mean(joint_loss) + eef_loss

            loss = action_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        return loss.item()
    
    def update(self, next_data, optim, stage=1, cur_step=0):
        
        obs_history, action, action_seq, next_obs = next_data
        obs_history = utils.to_torch(obs_history, device=self.device)
        next_obs    = utils.to_torch(next_obs, device=self.device)

        # print(obs_history[0].shape)
        # print(next_obs[0].shape)


        if stage == 1:
            return self.update_stage1(obs_history, next_obs, optim, cur_step=cur_step)
        if stage == 2:
            return self.update_stage2(obs_history, next_obs, optim, cur_step=cur_step)
        if stage == 3:
            action = torch.torch.as_tensor(action, device=self.device).float()
            return self.update_stage3(obs_history, next_obs, optim, action, cur_step=cur_step)


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

    if TRAIN:
        assert STAGE in [1, 2, 3, 4, 5], "Stage must be 1, 2, or 3."
        # Stage 4 = stage 1~2
        # Stage 5 = stage 1~3

        pretraining_data_dirs = []
        for task_id in range(10): 
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_goal']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs.append(offline_data_dir)
        eval_env = None

        if STAGE in [1, 2, 4, 5]:
        # Dataset Load

            replay_loader = make_replay_loader_dist(
                            replay_dir=pretraining_data_dirs, max_traj_per_task=50, max_size=100000000,
                            batch_size=64//1, num_workers=4,
                            save_snapshot=True, nstep=N_FUTURE_STEP, nstep_history=N_HISTORY, 
                            rank=0, world_size=1, n_code=10, vocab_size=200,
                            min_frequency=5, max_token_length=20)


            replay_dataset = iter(replay_loader)


        # Tensorboard Writer
        writer = SummaryWriter()

        if STAGE == 1 or STAGE == 4 or STAGE == 5:
            
            # Model, optimizer set
            m = MYMODEL(writer=writer).to(device=DEVICE)
            optim = torch.optim.Adam(m.parameters(), lr=0.001)

            # Stage 1 
            for i in tqdm(range(100010)):
                loss = m.update(next(replay_dataset), optim, stage=1, cur_step=i)
                writer.add_scalar("loss/stage1", loss, i)

                if i % 10000 == 0:
                    torch.save(m.state_dict(), s1_pt_name)
            torch.save(m.state_dict(), s1_pt_name)

        if STAGE == 2 or STAGE == 4 or STAGE == 5:
            # Model, optimizer set
            m = MYMODEL(writer=writer).to(device=DEVICE)
            optim = torch.optim.Adam(m.parameters(), lr=0.001)

            m.load_state_dict(torch.load(s1_pt_name))
            # Stage 2
            for i in tqdm(range(100010)):
                loss = m.update(next(replay_dataset), optim, stage=2, cur_step=i)
                writer.add_scalar("loss/stage2", loss, i)

                if i % 10000 == 0:
                    torch.save(m.state_dict(), s2_pt_name)
            torch.save(m.state_dict(), s2_pt_name)

        if STAGE == 3 or STAGE == 5:
            
            # Model, optimizer set
            m = MYMODEL(writer=writer).to(device=DEVICE)
            optim = torch.optim.Adam(m.parameters(), lr=0.001)

            if STAGE == 5:
                del replay_loader
                del replay_dataset
                
            for task_id in range(10): 
                print(task_id)
                s3_pt_name = s3_pt_prefix + "_task_{}.pt".format(task_id)
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict['libero_goal']()
                task = task_suite.get_task(task_id)
                task_name = task.name
                offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
                pretraining_data_dirs = [offline_data_dir, ]
                
                replay_loader = make_replay_loader_dist(
                                replay_dir=pretraining_data_dirs, max_traj_per_task=5, max_size=100000000,
                                batch_size=64//1, num_workers=4,
                                save_snapshot=True, nstep=N_FUTURE_STEP, nstep_history=N_HISTORY, 
                                rank=0, world_size=1, n_code=10, vocab_size=200,
                                min_frequency=5, max_token_length=20)
                replay_dataset = iter(replay_loader)
                eval_env = None

                m.load_state_dict(torch.load(s2_pt_name))

                for i in tqdm(range(25000)):
                    loss = m.update(next(replay_dataset), optim, stage=3, cur_step=i)
                    writer.add_scalar("loss/stage3_{}".format(task_id), loss, i)

                    if i % 10000 == 0:
                        torch.save(m.state_dict(), s3_pt_name)
                torch.save(m.state_dict(), s3_pt_name)

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
        
        m.update(next(replay_dataset), optim, stage=1)
        m.update(next(replay_dataset), optim, stage=2)
        m.update(next(replay_dataset), optim, stage=3)


        # writer.close()