import numpy as np
import argparse
import yaml

import torch

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='rlhf')
parser.add_argument('--linear', type=bool, default=True)
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--test_network', type=str, default='train_10')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

yaml_path = '../configs/' + args.env + '/' + args.reward + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

network_path = '../results/' + args.env + '/' + args.reward + '/' + args.test_network + '.pt'
reward_net = torch.load(network_path)
reward_net.eval()

probs = []
# compute probability of network on best state
for mushroom in config['best_mushroom']:
    mushroom = torch.Tensor(mushroom).to(args.device)
    pred_prob = torch.sigmoid(reward_net(mushroom)).cpu().detach().numpy()[0]

    probs.append(pred_prob)
    print('Probability: ', pred_prob)

print('Average probability: ', sum(probs)/len(probs))