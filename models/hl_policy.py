import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HL_BASE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 hidden_dim = 128,
                 output_dim = 128):
        super(HL_BASE, self).__init__()

        self.base_module = nn.Sequential(
            nn.Linear(feature_dim+task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.base_module(x)
    
class HL_PRED(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 hidden_dim = [512, 256, 128],
                 output_dim = 128):
        super(HL_PRED, self).__init__()

        assert len(hidden_dim)>=3

        module_seq = [nn.Linear(feature_dim+task_embed_dim, hidden_dim[0]), nn.GELU()]
        for i in range(1, len(hidden_dim)-1):
            module_seq.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            module_seq.append(nn.GELU())
        module_seq.append(nn.Linear(hidden_dim[-2], hidden_dim[-1]))
        module_seq.append(nn.GELU())

        reverse_seq = [nn.Linear(hidden_dim[-1], hidden_dim[-2]), nn.GELU()]
        for i in range(len(hidden_dim)-2, 1, -1):
            reverse_seq.append(nn.Linear(hidden_dim[i], hidden_dim[i-1]))
            reverse_seq.append(nn.GELU())
        reverse_seq.append(nn.Linear(hidden_dim[0], feature_dim))

        self.obs_module = nn.Sequential(*module_seq)
        self.pred_module = nn.Sequential(*reverse_seq)
        self.action_module = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.obs_module(x)
        return self.action_module(z)


    def forward_with_pred(self, x):
        z = self.obs_module(x)

        return self.action_module(z), self.pred_module(z)

class HL_PRED_MM(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 hidden_dim = [512, 256, 128],
                 output_dim = 128):
        super(HL_PRED_MM, self).__init__()

        assert len(hidden_dim)>=3

        module_seq = [nn.Linear(feature_dim+feature_dim+task_embed_dim, hidden_dim[0]), nn.GELU()]
        for i in range(1, len(hidden_dim)-1):
            module_seq.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            module_seq.append(nn.GELU())
        module_seq.append(nn.Linear(hidden_dim[-2], hidden_dim[-1]))
        module_seq.append(nn.GELU())

        reverse_seq = [nn.Linear(hidden_dim[-1], hidden_dim[-2]), nn.GELU()]
        for i in range(len(hidden_dim)-2, 1, -1):
            reverse_seq.append(nn.Linear(hidden_dim[i], hidden_dim[i-1]))
            reverse_seq.append(nn.GELU())
        reverse_seq.append(nn.Linear(hidden_dim[0], feature_dim+feature_dim))

        self.obs_module = nn.Sequential(*module_seq)
        self.pred_module = nn.Sequential(*reverse_seq)

        self.action_module = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.obs_module(x)
        return self.action_module(z)


    def forward_with_pred(self, x):
        z = self.obs_module(x)

        return self.action_module(z), self.pred_module(z)
    


class HL_PRED_MMS(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 hidden_dim = [512, 256, 128],
                 output_dim = 128):
        super(HL_PRED_MMS, self).__init__()

        assert len(hidden_dim)>=3

        module_seq = [nn.Linear(feature_dim+feature_dim+feature_dim+task_embed_dim, hidden_dim[0]), nn.GELU()]
        for i in range(1, len(hidden_dim)-1):
            module_seq.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            module_seq.append(nn.GELU())
        module_seq.append(nn.Linear(hidden_dim[-2], hidden_dim[-1]))
        module_seq.append(nn.GELU())

        reverse_seq = [nn.Linear(hidden_dim[-1], hidden_dim[-2]), nn.GELU()]
        for i in range(len(hidden_dim)-2, 1, -1):
            reverse_seq.append(nn.Linear(hidden_dim[i], hidden_dim[i-1]))
            reverse_seq.append(nn.GELU())
        reverse_seq.append(nn.Linear(hidden_dim[0], feature_dim+feature_dim+feature_dim))

        self.obs_module = nn.Sequential(*module_seq)
        self.pred_module = nn.Sequential(*reverse_seq)

        self.action_module = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Linear(hidden_dim[-1], output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.obs_module(x)
        return self.action_module(z)


    def forward_with_pred(self, x):
        z = self.obs_module(x)

        return self.action_module(z), self.pred_module(z)
    

class HL_PRED_AE(nn.Module):
    def __init__(self,
                 feature_dim = 128,
                 task_embed_dim = 768,
                 hidden_dim = [512, 256, 128],
                 action_hidden_dim = 512,
                 output_dim = 128):
        super(HL_PRED_AE, self).__init__()

        assert len(hidden_dim)>=3

        module_seq = [nn.Linear(feature_dim+task_embed_dim, hidden_dim[0]), nn.GELU()]
        for i in range(1, len(hidden_dim)-1):
            module_seq.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            module_seq.append(nn.GELU())
        module_seq.append(nn.Linear(hidden_dim[-2], hidden_dim[-1]))
        module_seq.append(nn.GELU())

        reverse_seq = [nn.Linear(hidden_dim[-1], hidden_dim[-2]), nn.GELU()]
        for i in range(len(hidden_dim)-2, 0, -1):
            reverse_seq.append(nn.Linear(hidden_dim[i], hidden_dim[i-1]))
            reverse_seq.append(nn.GELU())
        reverse_seq.append(nn.Linear(hidden_dim[0], feature_dim))

        self.obs_module = nn.Sequential(*module_seq)
        self.pred_module = nn.Sequential(*reverse_seq)
        self.action_module = nn.Sequential(
            nn.Linear(hidden_dim[-1], action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.obs_module(x)
        return self.action_module(z)


    def forward_with_pred(self, x):
        z = self.obs_module(x)

        return self.action_module(z), self.pred_module(z)
    

if __name__ == "__main__":
    device = "cuda:2"

    hl_policy = HL_PRED_AE(feature_dim=512, task_embed_dim=768, hidden_dim=[512, 480, 448, 416, 384, 352, 320, 288, 256], action_hidden_dim=512, output_dim=512).to(device)
    test_input = torch.randn(64, 512+768).to(device)

    a = hl_policy.forward_with_pred(test_input)