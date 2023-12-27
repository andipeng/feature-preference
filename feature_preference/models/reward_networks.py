import numpy as np
import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

# RLHF pairwise loss
class PairwiseLoss(nn.Module):
    def forward(self, preds1, preds2, prefs):
        outputs = (preds1 - preds2).view(-1)
        loss = -torch.mean(torch.log(torch.sigmoid(outputs * prefs.view(-1))))
        return loss

# reward network
class LinearRewardMLP(nn.Module):
    def __init__(
        self, state_dim):
        super().__init__()
        self.reward = nn.Linear(state_dim, 1)

    def forward(self, state):
        rew = self.reward(state)
        return rew
class RewardMLP(nn.Module):
    def __init__(
        self, state_dim):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        rew = self.reward(state)
        return rew

class FeaturePrefNetwork(nn.Module):
    def __init__(
        self, feature_dim, num_features):
        super().__init__()
        self.feat1 = nn.Linear(feature_dim, 1)
        self.feat2 = nn.Linear(feature_dim, 1)
        self.feat3 = nn.Linear(feature_dim, 1)
        self.feat4 = nn.Linear(feature_dim, 1)
        self.feat5 = nn.Linear(feature_dim, 1)
        self.feat6 = nn.Linear(feature_dim, 1)
        self.reward = nn.Linear(num_features, 1)

    def forward(self, state):
        # extracts all columns of each feature
        feat1 = self.feat1(state[:,0:3])
        feat2 = self.feat2(state[:,3:6])
        feat3 = self.feat3(state[:,6:9])
        feat4 = self.feat4(state[:,9:12])
        feat5 = self.feat5(state[:,12:15])
        feat6 = self.feat6(state[:,15:18])
        comp = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), dim=1)
        rew = self.reward(comp)
        return feat1, feat2, feat3, feat4, feat5, feat6, rew

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