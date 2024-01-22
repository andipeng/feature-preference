import numpy as np
import argparse
import yaml
import csv

import torch
from feature_preference.utils.mushroom_utils import calculate_best_mushroom
from feature_preference.utils.flight_utils import calculate_best_flight

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='feature_prefs_human') # rlhfs, feature_prefs, feature_prefs_human
parser.add_argument('--linear', type=bool, default=False)
parser.add_argument('--env', type=str, default='flights')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--test_network', type=str, default='train_1')
#parser.add_argument('--test_set', type=str, default='test_50')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

yaml_path = '../configs/' + args.env + '/' + args.reward + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

# Evaluate network on probability of gt reward

network_path = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.prefs_type + '_' + args.test_network + '.pt'
device = torch.device('cpu')
reward_net = torch.load(network_path, map_location=device)
reward_net.eval()

print("Evaluating " + args.test_network)
probs = []

if args.env == 'sim_mushrooms':
    best_states, max_reward = calculate_best_mushroom(config['features'], config['true_reward'])
elif args.env == 'flights':
    best_states = calculate_best_flight(config['true_reward'])

# compute probability of network on best state
for state in best_states:
    if args.env == 'sim_mushrooms':
        state = state.tolist()
    if args.prefs_type == 'rlhf':
        state = torch.Tensor(state).to(device)
        pred_prob = torch.sigmoid(reward_net(state)).cpu().detach().numpy()[0]
    else:
        state = torch.unsqueeze(torch.Tensor(state).to(device), dim=0)
        if args.env == 'sim_mushrooms':
            _, _, _, _, _, _, pred_prob = reward_net(state)
        elif args.env == 'flights':
            _, _, _, _, _, _, _, _, pred_prob = reward_net(state)
        pred_prob = torch.sigmoid(pred_prob).cpu().detach().numpy()[0][0]

    probs.append(pred_prob)
    print('Probability: ', pred_prob)

print('Average probability: ', sum(probs)/len(probs))