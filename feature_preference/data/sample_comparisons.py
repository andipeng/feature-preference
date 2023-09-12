import os
import numpy as np
import argparse
import pickle
import random
import csv
import itertools

##### Define features and actions
TEXTURES = [[1,0,0],[0,1,0],[0,0,1]]
COLORS = [[1,0,0],[0,1,0],[0,0,1]]
SHAPES = [[1,0,0],[0,1,0],[0,0,1]]
HEIGHTS = [[1,0,0],[0,1,0],[0,0,1]]
WEIGHTS = [[1,0,0],[0,1,0],[0,0,1]]
SMELLS = [[1,0,0],[0,1,0],[0,0,1]]
FEATURES = TEXTURES + COLORS + SHAPES + HEIGHTS + WEIGHTS + SMELLS

# Define all possible states
STATES = [{"texture": state[0], "color": state[1], "shape": state[2], "height": state[3], "weight": state[4], "smell": state[5]} 
           for state in itertools.product(TEXTURES, COLORS, SHAPES, HEIGHTS, WEIGHTS, SMELLS)]

TRUE_REWARDS = {"texture":[1,0,-1],
                "color":[1,0,-1],
                "shape":[1,0,-1],
                "height":[0,0,0],
                "weight":[0,0,0],
                "smell":[0,0,0]}

parser = argparse.ArgumentParser()
parser.add_argument('--num_comparisons', type=int, default=500)
parser.add_argument('--save_file', type=str, default='train_rewards500.csv')

args = parser.parse_args()

# turns dict state into flat state
def flatten_state(states):
    flattened_state = []
    for d in states:
        for value in d.values():
            flattened_state.append(value)
    return np.array(flattened_state).flatten()

# calculates state reward based on ground truth rewards
def calculate_reward(state):
    rewards = 0
    for feature in state.keys():
        index = state[feature].index(1)
        reward = TRUE_REWARDS[feature][index]
        rewards+=reward
    return rewards

print("========================================")
print("Sampling %d comparisons" % args.num_comparisons)
print("========================================")

# samples random mushroom from all possible features

with open(args.save_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    for sample in range(args.num_comparisons):
        final_list = []
        # samples 2 states and obtains rewards
        state1 = random.sample(STATES, 1)
        state2 = random.sample(STATES, 1)
        state1_reward = calculate_reward(state1[0])
        state2_reward = calculate_reward(state2[0])
        # flattens for writing
        state1 = flatten_state(state1)
        state2 = flatten_state(state2)
        # saves preferences
        pref = 1
        if state1_reward > state2_reward:
            pref = 0

        final_list.extend(state1)
        final_list.extend([state1_reward])
        final_list.extend(state2)
        final_list.extend([state2_reward])
        final_list.extend([pref])
        csv_writer.writerow(final_list)
