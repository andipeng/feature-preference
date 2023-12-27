import numpy as np
import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

# reward network
class LinearRewardMLP(nn.Module):
    def __init__(
        self, state_dim):
        super().__init__()
        self.reward = nn.Linear(state_dim, 1)

    def forward(self, state):
        rew = self.reward(state)
        return rew

# preference network
class PreferenceMLP(nn.Module):
    def __init__(
        self, state_dim):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pref = nn.Linear(2, 1)
        self.apply(weight_init)

    def forward(self, state1, state2):
        r1 = self.reward(state1)
        r2 = self.reward(state2)
        comp = torch.squeeze(torch.stack([r1,r2], dim=1))
        pref = self.pref(comp)
        return r1, r2, pref