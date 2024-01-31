import numpy as np
import argparse
import yaml
import csv

import torch
from feature_preference.utils.mushroom_utils import calculate_best_mushroom
from feature_preference.utils.flight_utils import calculate_best_flight

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='feature_prefs') # rlhf, feature_prefs, feature_prefs_human, rlhf_human, feature_prefs_gt
parser.add_argument('--linear', type=bool, default=False)
parser.add_argument('--env', type=str, default='flights')
parser.add_argument('--reward', type=str, default='reward3')
parser.add_argument('--test_network', type=str, default='train_10')
parser.add_argument('--test_set', type=str, default='test_50')
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

if args.env == 'sim_mushrooms' or args.env == 'user_study':
    best_states, max_reward = calculate_best_mushroom(config['features'], config['true_reward'])
elif args.env == 'flights':
    best_states = calculate_best_flight(config['true_reward'])

# compute probability of network on best state
for state in best_states:
    if args.env == 'sim_mushrooms' or args.env == 'user_study':
        state = state.tolist()
    if args.prefs_type == 'rlhf' or args.prefs_type == 'rlhf_human':
        state = torch.Tensor(state).to(device)
        pred_prob = torch.sigmoid(reward_net(state)).cpu().detach().numpy()[0]
    else:
        state = torch.unsqueeze(torch.Tensor(state).to(device), dim=0)
        if args.env == 'sim_mushrooms' or args.env == 'user_study':
            _, _, _, _, _, _, pred_prob = reward_net(state)
        elif args.env == 'flights':
            _, _, _, _, _, _, _, _, pred_prob = reward_net(state)
        pred_prob = torch.sigmoid(pred_prob).cpu().detach().numpy()[0][0]

    probs.append(pred_prob)
    print('Probability: ', pred_prob)

print('Average probability: ', sum(probs)/len(probs))

########################################################################

if args.env == 'sim_mushrooms':
    # Evaluate network on accuracy of predicting correct comparison out of test set pair
    print("\nAccuracy on test set: ")
    test_set_path = '../data/' + args.env + '/' + args.reward + '/' + args.test_set + '.csv'
    with open(test_set_path) as file_obj:
        reader_obj = csv.reader(file_obj)

        states1 = []
        states2 = []
        prefs = []
        for row in reader_obj:
            if args.env == 'sim_mushrooms' or args.env == 'user_study':
                states1.append(row[0:18])
                states2.append(row[19:37])
                prefs.append(row[38])
            elif args.env == 'flights':
                states1.append(row[0:8])
                states2.append(row[9:17])
                prefs.append(row[18])
        states1 = np.array(states1,dtype=float)
        states2 = np.array(states2,dtype=float)
        prefs = np.array(prefs,dtype=float)

    num_correct = 0.0
    for i in range(len(states1)):

        if args.prefs_type == 'rlhf' or args.prefs_type == 'rlhf_human':
            state1 = torch.Tensor(states1[i]).to(device)
            state2 = torch.Tensor(states2[i]).to(device)
            pred_prob1 = torch.sigmoid(reward_net(state1)).cpu().detach().numpy()[0]
            pred_prob2 = torch.sigmoid(reward_net(state2)).cpu().detach().numpy()[0]
        else:
            state1 = torch.unsqueeze(torch.Tensor(states1[i]).to(device), dim=0)
            state2 = torch.unsqueeze(torch.Tensor(states2[i]).to(device), dim=0)
            if args.env == 'sim_mushrooms' or args.env == 'user_study':
                _, _, _, _, _, _, pred_prob1 = reward_net(state1)
                _, _, _, _, _, _, pred_prob2 = reward_net(state2)
            elif args.env == 'flights':
                _, _, _, _, _, _, _, _, pred_prob1 = reward_net(state1)
                _, _, _, _, _, _, _, _, pred_prob2 = reward_net(state2)
            pred_prob1 = torch.sigmoid(pred_prob1).cpu().detach().numpy()[0][0]
            pred_prob2 = torch.sigmoid(pred_prob2).cpu().detach().numpy()[0][0]

        if pred_prob1 >= pred_prob2 and prefs[i] == 1:
            num_correct+=1
        elif pred_prob1 < pred_prob2 and prefs[i] == -1:
            num_correct+=1

    accuracy = num_correct / len(states1)
    print("Percent correct: %f \n" % accuracy)
