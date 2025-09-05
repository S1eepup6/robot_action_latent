import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MASKED_ATTENTION(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 total_step = 6,
                 hidden_dim = 128,
                 output_dim = 128,
                 device = "cpu"):
        super(MASKED_ATTENTION, self).__init__()

        self.total_step = total_step
        self.mask = self._build_mask().to(device)

        self.Q_linear = nn.Linear(feature_dim, hidden_dim)
        self.K_linear = nn.Linear(feature_dim, hidden_dim)
        self.V_linear = nn.Linear(feature_dim, hidden_dim)

        self.last_linears = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_mask(self):
        mask = torch.ones((self.total_step, self.total_step)).bool()
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(self, q, k, v):
        query = self.Q_linear(q)
        key = self.K_linear(k)
        key = torch.transpose(k, -2, -1)
        value = self.V_linear(v)

        qk = torch.matmul(query, key)
        qk.masked_fill_(self.mask, -math.inf)
        attn = F.softmax(qk, dim=-1)

        obs_z = torch.matmul(attn, value)

        obs_z = self.last_linears(obs_z)

        return obs_z
    

    
class ATTENTION_MODULE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 total_step = 6,
                 hidden_dim = 128,
                 output_dim = 128,
                 use_mask = True,
                 device = "cpu"):
        super(ATTENTION_MODULE, self).__init__()

        self.total_step = total_step
        self.use_mask = use_mask
        if self.use_mask:
            self.mask = self._build_mask().to(device)

        self.Q_linear = nn.Linear(feature_dim, hidden_dim)
        self.K_linear = nn.Linear(feature_dim, hidden_dim)
        self.V_linear = nn.Linear(feature_dim, hidden_dim)

        self.last_linears = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_mask(self):
        mask = torch.ones((self.total_step, self.total_step)).bool()
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(self, q, k, v):
        query = self.Q_linear(q)
        key = self.K_linear(k)
        key = torch.transpose(k, -2, -1)
        value = self.V_linear(v)

        qk = torch.matmul(query, key)
        if self.use_mask:
            qk.masked_fill_(self.mask, -math.inf)
        attn = F.softmax(qk, dim=-1)

        obs_z = torch.matmul(attn, value)

        obs_z = self.last_linears(obs_z)

        return obs_z
    
        

if __name__ == "__main__":
    test_model = MASKED_ATTENTION()
    test_input = torch.randn(64, 6, 128)
    test_task = torch.randn(64, 768)

    output = test_model(test_input, test_input, test_input)
    print(output.shape)