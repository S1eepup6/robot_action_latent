import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.masked_attention import MASKED_ATTENTION, ATTENTION_MODULE

class FDM(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 time_step = 6,
                 hidden_dim = 128,
                 output_dim = 128):
        super(FDM, self).__init__()

        self.obs_only_module = nn.Sequential(
            nn.Linear(feature_dim*time_step+feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim*time_step),
            nn.GELU(),
        )


        self.obs_module = nn.Sequential(
            nn.Linear(feature_dim*time_step+feature_dim, hidden_dim),
            nn.GELU(),
        )

        self.task_module = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim*time_step),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.obs_only_module(x)
    
    def forward_task(self, obs, task):
        obs_z = self.obs_module(obs)
        task_z = self.task_module(task)
        total_z = torch.concat([obs_z, task_z], dim=-1)

        return self.total_module(total_z)
    

class FDM_ONE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 latent_dim = 128,
                 task_embed_dim = 128,
                 time_step = 1,
                 hidden_dim = 128):
        super(FDM_ONE, self).__init__()

        self.linear_module = nn.Sequential(
            nn.Linear(feature_dim*time_step+latent_dim+task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        return self.linear_module(x)
    

class FDM_MASKED_ATTN(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 time_step = 6,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_MASKED_ATTN, self).__init__()

        self.total_step = time_step
        self.mask = self._build_mask().to(device)

        self.Q_linear = nn.Linear(feature_dim, hidden_dim)
        self.K_linear = nn.Linear(feature_dim, hidden_dim)
        self.V_linear = nn.Linear(feature_dim, hidden_dim)

        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*time_step, hidden_dim),
            nn.GELU(),
        )

        self.latent_module = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        self.task_module = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim*time_step),
            nn.ReLU(),
        )

    def _build_mask(self):
        mask = torch.ones((self.total_step, self.total_step)).bool()
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(self, x, z_q, task):
        query = self.Q_linear(x)
        key = self.K_linear(x)
        key = torch.transpose(x, -2, -1)
        value = self.V_linear(x)

        qk = torch.matmul(query, key)
        qk.masked_fill_(self.mask, -math.inf)
        attn = F.softmax(qk, dim=-1)

        obs_z = torch.matmul(attn, value).flatten(start_dim=1)

        obs_z = self.obs_module(obs_z)
        task_z = self.task_module(task)
        latent_z = self.latent_module(z_q)
        total_z = torch.concat([obs_z, latent_z, task_z], dim=-1)

        return self.total_module(total_z)
    
    
class FDM_MASKED_ATTN_FUTURE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 time_step = 6,
                 future_step = 1,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_MASKED_ATTN_FUTURE, self).__init__()

        self.total_step = time_step
        self.future_step = future_step
        self.mask = self._build_mask().to(device)

        self.Q_linear = nn.Linear(feature_dim, hidden_dim)
        self.K_linear = nn.Linear(feature_dim, hidden_dim)
        self.V_linear = nn.Linear(feature_dim, hidden_dim)

        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*time_step, hidden_dim),
            nn.GELU(),
        )

        self.latent_module = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        self.task_module = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim*future_step),
            nn.ReLU(),
        )

    def _build_mask(self):
        mask = torch.ones((self.total_step, self.total_step)).bool()
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(self, x, z_q, task):
        query = self.Q_linear(x)
        key = self.K_linear(x)
        key = torch.transpose(x, -2, -1)
        value = self.V_linear(x)

        qk = torch.matmul(query, key)
        qk.masked_fill_(self.mask, -math.inf)
        attn = F.softmax(qk, dim=-1)

        obs_z = torch.matmul(attn, value).flatten(start_dim=1)

        obs_z = self.obs_module(obs_z)
        task_z = self.task_module(task)
        latent_z = self.latent_module(z_q)
        total_z = torch.concat([obs_z, latent_z, task_z], dim=-1)

        return self.total_module(total_z)

class FDM_ATTN_2(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 future_step = 5,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_ATTN_2, self).__init__()

        self.total_step = total_step
        self.future_step = future_step
        self.task_embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.obs_attention_layer = MASKED_ATTENTION(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, device=device).to(device)
        self.task_attention_layer = MASKED_ATTENTION(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, device=device).to(device)
        
        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*total_step, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim*self.future_step),
            nn.ReLU(),
        )

    def forward(self, x, z_q):
        a_x = self.obs_attention_layer(x, x, x)

        task_z = self.task_embed(z_q)
        task_z_seq = task_z.unsqueeze(-2).repeat(1, self.total_step, 1)

        z = self.task_attention_layer(a_x, task_z_seq, a_x).flatten(start_dim=1)

        z = self.obs_module(z)

        total_z = torch.concat([z, task_z], dim=-1)

        return self.total_module(total_z)
        


class FDM_ATTN_2_TASK(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_ATTN_2_TASK, self).__init__()

        self.total_step = total_step
        self.task_embed = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.z_embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.obs_attention_layer = MASKED_ATTENTION(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, device=device).to(device)
        self.task_attention_layer = MASKED_ATTENTION(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, device=device).to(device)
        self.z_attention_layer = MASKED_ATTENTION(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, device=device).to(device)
        
        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*total_step, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(feature_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim*self.total_step),
            nn.ReLU(),
        )

    def forward(self, x, z_q, task):
        a_x = self.obs_attention_layer(x, x, x)

        
        task_z = self.task_embed(task)
        task_z_seq = task_z.unsqueeze(-2).repeat(1, self.total_step, 1)
        
        t_x = self.z_attention_layer(a_x, task_z_seq, a_x)

        z_embed = self.z_embed(z_q)
        z_embed_seq = z_embed.unsqueeze(-2).repeat(1, self.total_step, 1)

        z = self.z_attention_layer(t_x, z_embed_seq, t_x).flatten(start_dim=1)

        z = self.obs_module(z)

        total_z = torch.concat([z, z_embed, task_z], dim=-1)

        return self.total_module(total_z)
        
        
class FDM_MASK_SEL(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 future_step = 5,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_MASK_SEL, self).__init__()

        self.total_step = total_step
        self.future_step = future_step
        self.task_embed = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.z_embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.obs_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=True, device=device).to(device)
        self.task_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=False, device=device).to(device)
        self.z_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=False, device=device).to(device)
        
        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*total_step, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(feature_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim*self.future_step)
        )

    def forward(self, x, z_q, task):
        a_x = self.obs_attention_layer(x, x, x)
        
        task_z = self.task_embed(task)
        task_z_seq = task_z.unsqueeze(-2).repeat(1, self.total_step, 1)
        
        t_x = self.task_attention_layer(a_x, task_z_seq, a_x)

        z_embed = self.z_embed(z_q)
        z_embed_seq = z_embed.unsqueeze(-2).repeat(1, self.total_step, 1)

        z = self.z_attention_layer(t_x, z_embed_seq, t_x).flatten(start_dim=1)

        z = self.obs_module(z)

        total_z = torch.concat([z, z_embed, task_z], dim=-1)

        return self.total_module(total_z)
    
class FDM_ALL_MASK(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 total_step = 5,
                 future_step = 5,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(FDM_ALL_MASK, self).__init__()

        self.total_step = total_step
        self.future_step = future_step
        self.task_embed = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.z_embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.obs_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=True, device=device).to(device)
        self.task_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=True, device=device).to(device)
        self.z_attention_layer = ATTENTION_MODULE(feature_dim=feature_dim, total_step=total_step, hidden_dim=hidden_dim, output_dim=feature_dim, use_mask=True, device=device).to(device)
        
        self.obs_module = nn.Sequential(
            nn.Linear(hidden_dim*total_step, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

        self.total_module = nn.Sequential(
            nn.Linear(feature_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim*self.future_step)
        )

    def forward(self, x, z_q, task):
        a_x = self.obs_attention_layer(x, x, x)
        
        task_z = self.task_embed(task)
        task_z_seq = task_z.unsqueeze(-2).repeat(1, self.total_step, 1)
        
        t_x = self.task_attention_layer(a_x, task_z_seq, a_x)

        z_embed = self.z_embed(z_q)
        z_embed_seq = z_embed.unsqueeze(-2).repeat(1, self.total_step, 1)

        z = self.z_attention_layer(t_x, z_embed_seq, t_x).flatten(start_dim=1)

        z = self.obs_module(z)

        total_z = torch.concat([z, z_embed, task_z], dim=-1)

        return self.total_module(total_z)
    
if __name__ == "__main__":
    test_model = FDM_MASKED_ATTN()
    test_input = torch.randn(64, 6, 128)
    test_task = torch.randn(64, 768)

    output = test_model(test_input, test_task)
    print(output.shape)