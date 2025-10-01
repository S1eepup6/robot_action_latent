import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEER_ATTN_MODULE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 total_step = 6,
                 future_step = 1,
                 multi_modality = 2,
                 hidden_dim = 128,
                 output_dim = 128,
                 use_mask = True,
                 device = "cpu"):
        super(SEER_ATTN_MODULE, self).__init__()

        self.total_step = total_step
        self.future_step = future_step
        self.multi_modality = multi_modality
        self.image_block = total_step * multi_modality
        self.future_block = future_step * multi_modality

        self.use_mask = use_mask
        if self.use_mask:
            self.mask = self._build_mask(addition=2).to(device)

        self.norm = nn.LayerNorm(feature_dim)

        self.Q_linear = nn.Linear(feature_dim, hidden_dim)
        self.K_linear = nn.Linear(feature_dim, hidden_dim)
        self.V_linear = nn.Linear(feature_dim, hidden_dim)

        self.linear_norm = nn.LayerNorm(hidden_dim)
        self.last_linears = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_mask(self, addition=0):

        mask = torch.ones((self.total_step, self.total_step))
        mask = torch.triu(mask, diagonal=1)

        mask = torch.concat([mask] * self.multi_modality, dim=0)
        mask = torch.concat([mask] * self.multi_modality, dim=1)

        if addition == 0:
            return mask.bool()
        mask_right = torch.ones((self.image_block, 2 + self.future_block))
        mask_upper = torch.concat([mask, mask_right], dim=1)

        mask_lower = torch.zeros((2 + self.future_block, self.image_block + 2 + self.future_block))
        mask = torch.concat([mask_upper, mask_lower], dim=0)

        return mask.bool()


    def forward(self, x):
        x = self.norm(x)

        query = self.Q_linear(x)
        key = self.K_linear(x)
        key = torch.transpose(x, -2, -1)
        value = self.V_linear(x)

        qk = torch.matmul(query, key)
        if self.use_mask:
            qk.masked_fill_(self.mask, -math.inf)
        attn = F.softmax(qk, dim=-1)

        res = torch.matmul(attn, value)

        res = self.linear_norm(res)
        res = self.last_linears(res)

        return res

    
class SEER_ENCODER(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 future_step = 5,
                 multi_modality = 2,
                 attn_depth = 3,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(SEER_ENCODER, self).__init__()

        assert attn_depth >= 3

        self.total_step = total_step
        self.future_step = future_step
        self.hidden_dim = hidden_dim
        self.image_block = self.total_step * multi_modality
        self.multi_modality = multi_modality

        self.task_embed = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.action_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pred_token = nn.Parameter(torch.zeros(1, self.future_step * self.multi_modality, self.hidden_dim))

        self.attn_layers = []

        self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=feature_dim, total_step=total_step, future_step=future_step, multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=hidden_dim, use_mask=True, device=device).to(device))
        for i in range(attn_depth-2): 
            self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=hidden_dim, total_step=total_step, future_step=future_step,  multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=hidden_dim, use_mask=True, device=device).to(device))
        self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=hidden_dim, total_step=total_step, future_step=future_step,  multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=output_dim, use_mask=True, device=device).to(device))


        self.attn_seq_layer = nn.Sequential(*self.attn_layers)

    def forward(self, obs, obs_wrist, task):
        B, T, H = obs.shape
        task_z = self.task_embed(task)
        action_token = self.action_token.repeat(B, 1, 1)
        pred_token = self.pred_token.repeat(B, 1, 1)

        attn_z = torch.concat([obs, obs_wrist, task_z, action_token, pred_token], dim=-2)
        
        attn_z = self.attn_seq_layer.forward(attn_z)

        action_z = attn_z[:, -(self.future_step*self.multi_modality+1)]
        pred_z = attn_z[:, -self.future_step*self.multi_modality:-self.future_step]
        pred_z_wrist = attn_z[:, -self.future_step:]

        return action_z, pred_z, pred_z_wrist
    
    
    
class SEER_ENCODER_STATE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 future_step = 5,
                 multi_modality = 3,
                 attn_depth = 3,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(SEER_ENCODER_STATE, self).__init__()

        assert attn_depth >= 3

        self.total_step = total_step
        self.future_step = future_step
        self.hidden_dim = hidden_dim
        self.image_block = self.total_step * multi_modality
        self.multi_modality = multi_modality

        self.task_embed = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.action_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pred_token = nn.Parameter(torch.zeros(1, self.future_step * self.multi_modality, self.hidden_dim))

        self.attn_layers = []

        self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=feature_dim, total_step=total_step, future_step=future_step, multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=hidden_dim, use_mask=True, device=device).to(device))
        for i in range(attn_depth-2): 
            self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=hidden_dim, total_step=total_step, future_step=future_step,  multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=hidden_dim, use_mask=True, device=device).to(device))
        self.attn_layers.append(SEER_ATTN_MODULE(feature_dim=hidden_dim, total_step=total_step, future_step=future_step,  multi_modality=multi_modality, hidden_dim=hidden_dim, output_dim=output_dim, use_mask=True, device=device).to(device))


        self.attn_seq_layer = nn.Sequential(*self.attn_layers)

    def forward(self, obs, obs_wrist, state, task):
        B, T, H = obs.shape
        task_z = self.task_embed(task)
        action_token = self.action_token.repeat(B, 1, 1)
        pred_token = self.pred_token.repeat(B, 1, 1)

        attn_z = torch.concat([obs, obs_wrist, state, task_z, action_token, pred_token], dim=-2)
        
        attn_z = self.attn_seq_layer.forward(attn_z)

        action_z = attn_z[:, -((self.future_step*3)+1)]
        pred_z = attn_z[:, -self.future_step*3:-self.future_step*2]
        pred_z_wrist = attn_z[:, -self.future_step*2:-self.future_step]
        pred_z_state = attn_z[:, -self.future_step:]

        return action_z, pred_z, pred_z_wrist, pred_z_state
    


if __name__ == "__main__":
    device = 'cuda:2'
    test_module = SEER_ENCODER(device=device).to(device)

    test_input_obs = torch.Tensor(np.random.randn(1, 5, 128)).float().to(device)
    test_input_obs_wrist = torch.Tensor(np.random.randn(1, 5, 128)).float().to(device)
    test_input_task = torch.Tensor(np.random.randn(1, 1, 768)).float().to(device)
    
    test_output = test_module(test_input_obs, test_input_obs_wrist, test_input_task)
    print(test_output[0].shape)
    print(test_output[1].shape)
    print(test_output[2].shape)
    # print(test_output[3].shape)