import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DECODER(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim+feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.base_module(x)
    
class DECODER_L(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_L, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim+feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.joint_actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        self.eef_selector = nn.Sequential(
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        l = self.base_module(x)
        return self.joint_actor(l), self.eef_selector(l)
    
class DECODER_L_MM(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 multi_modality = 2,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_L_MM, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim*multi_modality + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.joint_actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        self.eef_selector = nn.Sequential(
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        l = self.base_module(x)
        return self.joint_actor(l), self.eef_selector(l)


class DECODER_MM(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 multi_modality = 2,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_MM, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim*multi_modality + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.joint_actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.eef_selector = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        l = self.base_module(x)
        return self.joint_actor(l), self.eef_selector(l).squeeze(dim=-1)

class DECODER_Z(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_Z, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.base_module(x)
    

class DECODER_SEQ(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 hidden_dim = 128,
                 output_dim = 128,
                 output_len = 4):
        super(DECODER_SEQ, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim*output_len)
        )

    def forward(self, x):
        return self.base_module(x)
    
class DECODER_PRED(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 hidden_dim = 128,
                 output_dim = 128):
        super(DECODER_PRED, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim+feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_module = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.pred_module = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.base_module(x)
        a = self.action_module(z)
        p = self.pred_module(z)

        return a, p