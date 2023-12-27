import numpy as np
import argparse
import yaml
import csv

import torch

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='rlhf')
parser.add_argument('--linear', type=bool, default=True)
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--test_network', type=str, default='train_100')
parser.add_argument('--test_set', type=str, default='test_50')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

yaml_path = '../configs/' + args.env + '/' + args.reward + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

# Evaluate network on probability of gt reward

network_path = '../results/' + args.env + '/' + args.reward + '/' + args.test_network + '.pt'
reward_net = torch.load(network_path)
reward_net.eval()

print("Evaluating " + args.test_network)
probs = []
# compute probability of network on best state
for mushroom in config['best_mushroom']:
    mushroom = torch.Tensor(mushroom).to(args.device)
    pred_prob = torch.sigmoid(reward_net(mushroom)).cpu().detach().numpy()[0]

    probs.append(pred_prob)
    print('Probability: ', pred_prob)

print('Average probability: ', sum(probs)/len(probs))

########################################################################

# Evaluate network on accuracy of predicting correct comparison out of test set pair
print("\nEvaluating accuracy on test set: ")
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
    mushroom1 = torch.Tensor(states1[i]).to(args.device)
    mushroom2 = torch.Tensor(states2[i]).to(args.device)
    pred_r1 = torch.sigmoid(reward_net(mushroom1)).cpu().detach().numpy()[0]
    pred_r2 = torch.sigmoid(reward_net(mushroom2)).cpu().detach().numpy()[0]

    if pred_r1 > pred_r2 and prefs[i] == 1:
        num_correct+=1
    elif pred_r1 <= pred_r2 and prefs[i] == -1:
        num_correct+=1

accuracy = num_correct / len(states1)
print("Percent correct: %f \n" % accuracy)