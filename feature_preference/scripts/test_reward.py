import numpy as np
import argparse
import yaml
import csv

import torch
from feature_preference.utils.mushroom_utils import calculate_best_mushroom

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='rlhf') # rlhfs or feature_prefs
parser.add_argument('--linear', type=bool, default=False)
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward3')
parser.add_argument('--test_network', type=str, default='train_5')
parser.add_argument('--test_set', type=str, default='test_50')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

yaml_path = '../configs/' + args.env + '/' + args.reward + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

# Evaluate network on probability of gt reward

network_path = '../results/' + args.env + '/' + args.reward + '/' + args.prefs_type + '_' + args.test_network + '.pt'
reward_net = torch.load(network_path)
reward_net.eval()

print("Evaluating " + args.test_network)
probs = []

best_mushrooms, max_reward = calculate_best_mushroom(config['features'], config['true_reward'])
# compute probability of network on best state
for mushroom in best_mushrooms:

    if args.prefs_type == 'rlhf':
        mushroom = torch.Tensor(mushroom.tolist()).to(args.device)
        pred_prob = torch.sigmoid(reward_net(mushroom)).cpu().detach().numpy()[0]
    elif args.prefs_type == 'feature_prefs':
        mushroom = torch.unsqueeze(torch.Tensor(mushroom.tolist()).to(args.device), dim=0)
        _, _, _, _, _, _, pred_prob = reward_net(mushroom)
        pred_prob = torch.sigmoid(pred_prob).cpu().detach().numpy()[0][0]

    probs.append(pred_prob)
    print('Probability: ', pred_prob)

print('Average probability: ', sum(probs)/len(probs))

########################################################################

# Evaluate network on accuracy of predicting correct comparison out of test set pair
print("\nAccuracy on test set: ")
test_set_path = '../data/' + args.env + '/' + args.reward + '/' + args.test_set + '.csv'
with open(test_set_path) as file_obj:
    reader_obj = csv.reader(file_obj)

    states1 = []
    states2 = []
    prefs = []
    for row in reader_obj:
        states1.append(row[0:18])
        states2.append(row[19:37])
        prefs.append(row[38])
    states1 = np.array(states1,dtype=int)
    states2 = np.array(states2,dtype=int)
    prefs = np.array(prefs,dtype=int)

num_correct = 0.0
for i in range(len(states1)):

    if args.prefs_type == 'rlhf':
        mushroom1 = torch.Tensor(states1[i]).to(args.device)
        mushroom2 = torch.Tensor(states2[i]).to(args.device)
        pred_prob1 = torch.sigmoid(reward_net(mushroom1)).cpu().detach().numpy()[0]
        pred_prob2 = torch.sigmoid(reward_net(mushroom2)).cpu().detach().numpy()[0]
    elif args.prefs_type == 'feature_prefs':
        mushroom1 = torch.unsqueeze(torch.Tensor(states1[i]).to(args.device), dim=0)
        mushroom2 = torch.unsqueeze(torch.Tensor(states2[i]).to(args.device), dim=0)
        _, _, _, _, _, _, pred_prob1 = reward_net(mushroom1)
        _, _, _, _, _, _, pred_prob2 = reward_net(mushroom2)
        pred_prob1 = torch.sigmoid(pred_prob1).cpu().detach().numpy()[0][0]
        pred_prob2 = torch.sigmoid(pred_prob2).cpu().detach().numpy()[0][0]

    if pred_prob1 >= pred_prob2 and prefs[i] == 1:
        num_correct+=1
    elif pred_prob1 < pred_prob2 and prefs[i] == -1:
        num_correct+=1

accuracy = num_correct / len(states1)
print("Percent correct: %f \n" % accuracy)