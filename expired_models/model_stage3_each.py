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

DEVICE = 'cuda:1'

def construct_task_data_path(root_dir, task_name, task_data_dir_suffix='framestack1'):
    return Path(root_dir) / (task_name.lower()+('' if not task_data_dir_suffix or task_data_dir_suffix == 'None' else task_data_dir_suffix))

def replay_iter(replay_loader):
    return iter(replay_loader)

class MYMODEL(nn.Module):
    def __init__(self,
                 time_step=5,
                 future_step=1,
                 obs_shape=[3, 128, 128],
                 n_code=32, 
                 feature_dim=128, 
                 hidden_dim=128, 
                 output_dim=128,
                 task_embed_dim=768,
                 action_dim=7,
                 writer=None
                 ):
        super(MYMODEL, self).__init__()

        self.time_step = time_step
        self.future_step = future_step
        self.total_step = time_step + future_step
        self.writer = writer

        self.encoder = ResnetEncoder(input_shape=obs_shape, output_size=feature_dim).to(DEVICE)
        self.idm = nn.Sequential(
            nn.Linear(feature_dim*self.total_step, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.a_quantizer = VectorQuantizer(n_code, feature_dim, device=DEVICE)
        self.fdm = nn.Sequential(
            nn.Linear(feature_dim*self.time_step+feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim*time_step),
            nn.ReLU(),
        )

        self.hl_policy = nn.Sequential(
            nn.Linear(feature_dim+task_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim+feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder_head = GMMHead(hidden_dim, action_dim, hidden_dim, loss_coef=0.01)


    def update_stage1(self, obs_history, next_obs, optim, cur_step=0):
        '''
        Observations -> idm -> a_quantizer -> q_emb
        q_emb + start_obs -> fdm
        fdm ---- observations[start+1:]
        '''

        ### (batch_size, num_step_history, 3, 128, 128)
        obs_agent_history, obs_wrist_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_task_embedding = next_obs

        obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        next_time_step = next_obs_agent.shape[1]

        # task_embedding_history = torch.concat((task_embedding_history, next_task_embedding), dim=1)

        # print(idm_obs_agent.shape)
        # print(obs_wrist_history.shape)
        # print(task_embedding_history.shape)

        task_embedding_history = task_embedding_history.reshape(-1, task_embedding_history.shape[-1]).float()

        obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
        obs_embed = self.encoder.forward(obs_full_history)
        obs_embed = obs_embed.reshape(batch_size, time_step+next_time_step, -1)

        fdm_embed = obs_embed[:, :-next_time_step].flatten(start_dim=1)

        z = self.idm.forward(obs_embed.flatten(start_dim=1))
        q_loss, z_q, _, _, _ = self.a_quantizer.forward(z)

        fdm_input = torch.concat([fdm_embed, z_q], dim=-1)
        fdm_output = self.fdm.forward(fdm_input)
        fdm_output = fdm_output.reshape(batch_size, time_step, -1)

        ans = obs_embed[:, 1:]

        loss = F.mse_loss(fdm_output, ans) + q_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if self.writer is not None:
            self.writer.add_scalar("loss/quant", q_loss, cur_step)
            self.writer.add_scalar("loss/fdm", loss.item(), cur_step)
        else:
            print("q_loss {},      fdm loss {}".format(q_loss, loss.item()))

        return q_loss, loss.item()


    def update_stage2(self, obs_history, next_obs, optim, cur_step=0):
        '''
        Observations -> idm(frozen) -> a_quantizer(frozen) -> skill
        cur_obs + task_emb -> hl_policy -> a_quantizer(frozen) ---- skill
        '''
        
        ### (batch_size, num_step_history, 3, 128, 128)
        obs_agent_history, obs_wrist_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_task_embedding = next_obs

        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        
        with torch.no_grad():
            obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
            obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed = self.encoder.forward(obs_full_history)
            obs_embed = obs_embed.reshape(batch_size, time_step+self.future_step, -1)

            z = self.idm.forward(obs_embed.flatten(start_dim=1))


        cur_obs = obs_embed[:, -1].flatten(start_dim=1)
        cur_task = task_embedding_history[:, -1]
        hl_policy_input = torch.concat([cur_obs, cur_task], dim=1).float()

        z_policy = self.hl_policy(hl_policy_input)   

        loss = F.mse_loss(z_policy, z)

        if self.writer is not None:
            self.writer.add_scalar("loss/stage2", loss.item(), cur_step)

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
        obs_agent_history, obs_wrist_history, task_embedding_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        next_obs_agent, next_obs_wrist, next_task_embedding = next_obs

        batch_size, time_step, num_channel, img_size = obs_agent_history.shape[0], obs_agent_history.shape[1], obs_agent_history.shape[2], obs_agent_history.shape[3]
        

        with torch.no_grad():
            obs_steps = []
            obs_full_history = torch.concat((obs_agent_history, next_obs_agent), dim=1)
            obs_full_history = obs_full_history.reshape(-1, num_channel, img_size, img_size).float()
            obs_embed = self.encoder.forward(obs_full_history)
            obs_embed = obs_embed.reshape(batch_size, time_step+self.future_step, -1)

            first_obs = obs_embed[:, 0].flatten(start_dim=1)
            cur_task = task_embedding_history[:, 0]
            
            for i in range(self.time_step):
                obs_steps.append(obs_embed[:, i].flatten(start_dim=1))

            hl_policy_input = torch.concat([first_obs, cur_task], dim=1).float()



        for cur_obs in obs_steps:
            with torch.no_grad():
                z_policy = self.hl_policy(hl_policy_input)
                q_loss, z_q, _, _, _ = self.a_quantizer.forward(z_policy)

            decoder_input = torch.concat([cur_obs, z_q], dim=1)

            action = self.decoder(decoder_input)
            action = self.decoder_head(action)

            loss = self.decoder_head.loss_fn(action, ground_truth_action, reduction='none')
            loss = torch.mean(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if self.writer is not None:
            self.writer.add_scalar("loss/stage3", loss.item(), cur_step)

        return loss.item()
    
    def update(self, next_data, optim, stage=1, cur_step=0):
        
        obs_history, action, action_seq, next_obs = next_data
        obs_history = utils.to_torch(obs_history, device=DEVICE)
        next_obs    = utils.to_torch(next_obs, device=DEVICE)

        # print(obs_history[0].shape)
        # print(next_obs[0].shape)


        if stage == 1:
            return self.update_stage1(obs_history, next_obs, optim, cur_step=cur_step)
        if stage == 2:
            return self.update_stage2(obs_history, next_obs, optim, cur_step=cur_step)
        if stage == 3:
            action = torch.torch.as_tensor(action, device=DEVICE).float()
            return self.update_stage3(obs_history, next_obs, optim, action, cur_step=cur_step)


if __name__ == "__main__":
    TRAIN = True
    SEED = 0

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

        # Tensorboard Writer
        writer = None

        # Model, optimizer set
        m = MYMODEL(writer=writer).to(device=DEVICE)
        optim = torch.optim.Adam(m.parameters(), lr=0.001)

        s2_pt_name = "/data/libero/exp_results/s2_06-09_12:59_0.pt"

        for task_id in range(22, 30): 
            s3_pt_name = "/data/libero/exp_each/s3_06-09_12:59_0_task_{}.pt".format(task_id)

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_90']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs = [offline_data_dir, ]
            
            replay_loader = make_replay_loader_dist(
                            replay_dir=pretraining_data_dirs, max_traj_per_task=5, max_size=100000000,
                            batch_size=64//1, num_workers=4,
                            save_snapshot=True, nstep=1, nstep_history=5, 
                            rank=0, world_size=1, n_code=10, vocab_size=200,
                            min_frequency=5, max_token_length=20)
            replay_dataset = iter(replay_loader)
            eval_env = None

            m.load_state_dict(torch.load(s2_pt_name))

            for i in tqdm(range(30001)):
                m.update(next(replay_dataset), optim, stage=3, cur_step=i)
                if i % 10000 == 0:
                    torch.save(m.state_dict(), s3_pt_name)
            torch.save(m.state_dict(), s3_pt_name)

############################### Code Test ###############################
    else:
        # writer = SummaryWriter()

        m = MYMODEL().to(device=DEVICE)
        optim = torch.optim.Adam(m.parameters(), lr=0.001)

        pretraining_data_dirs = []
        for task_id in range(1): 
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict['libero_90']()
            task = task_suite.get_task(task_id)
            task_name = task.name
            offline_data_dir = construct_task_data_path(DATA_STORAGE_DIR, task_name, TASK_DATA_DIR_SUFFIX)
            pretraining_data_dirs.append(offline_data_dir)
        eval_env = None
        replay_loader = make_replay_loader_dist(
                        replay_dir=pretraining_data_dirs, max_traj_per_task=10, max_size=10000000,
                        batch_size=64//1, num_workers=4,
                        save_snapshot=True, nstep=1, nstep_history=5, 
                        rank=0, world_size=1, n_code=10, vocab_size=200,
                        min_frequency=5, max_token_length=20)
        replay_dataset = iter(replay_loader)
        
        m.update(next(replay_dataset), optim, stage=3)


        # writer.close()