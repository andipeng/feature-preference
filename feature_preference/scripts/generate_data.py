import numpy as np
import yaml
import argparse
import random
import csv

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--config', type=str, default='reward1')
parser.add_argument('--num_comparisons', type=int, default=100)
parser.add_argument('--test', type=bool, default=True)

args = parser.parse_args()

yaml_path = '../configs/' + args.env + '/' + args.config + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

def sample_state(features):
    state = {}
    for feature in features:
        sampled_feature = random.sample(features[feature], 1)
        state[feature] = sampled_feature[0]
    return [state]

# turns dict state into flat state
def flatten_state(states):
    flattened_state = []
    for d in states:
        for value in d.values():
            flattened_state.append(value)
    return np.array(flattened_state).flatten()

# calculates state reward based on ground truth rewards
def calculate_reward(state, true_reward):
    rewards = 0
    for feature in state.keys():
        index = state[feature].index(1)
        reward = true_reward[feature][index]
        rewards+=reward
    return rewards

def calculate_feature_prefs(state1, state2, true_reward):
    feature_prefs = []
    for feature in state1.keys():
        index1 = state1[feature].index(1)
        index2 = state2[feature].index(1)
        reward1 = true_reward[feature][index1]
        reward2 = true_reward[feature][index2]
        
        feature_pref = 1
        if reward1 > reward2:
            feature_pref = 0
        feature_prefs.append(feature_pref)
    return feature_prefs

print("========================================")
print("Sampling %d comparisons" % args.num_comparisons)
print("========================================")

if args.test:
    path_out = '../data/' + args.env + '/' + args.config + '/test_' + str(args.num_comparisons) + '.csv'
else:
    path_out = '../data/' + args.env + '/' + args.config + '/train_' + str(args.num_comparisons) + '.csv'
# samples random mushroom from all possible features

with open(path_out, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    for sample in range(args.num_comparisons):
        final_list = []
        # samples 2 states and obtains rewards
        state1 = sample_state(config['features'])
        state2 = sample_state(config['features'])
        state1_reward = calculate_reward(state1[0], config['true_reward'])
        state2_reward = calculate_reward(state2[0], config['true_reward'])
        feature_prefs = calculate_feature_prefs(state1[0], state2[0], config['true_reward'])
        # flattens for writing
        state1 = flatten_state(state1)
        state2 = flatten_state(state2)
        # saves preferences
        pref = 1
        if state2_reward > state1_reward:
            pref = -1

        final_list.extend(state1)
        final_list.extend([state1_reward])
        final_list.extend(state2)
        final_list.extend([state2_reward])
        final_list.extend([pref])
        final_list.extend(feature_prefs)
        csv_writer.writerow(final_list)
