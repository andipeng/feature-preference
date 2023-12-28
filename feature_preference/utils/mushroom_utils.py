import random
import numpy as np
import itertools

def sample_state(features):
    state = {}
    for feature in features:
        sampled_feature = random.sample(features[feature], 1)
        state[feature] = sampled_feature[0]
    return [state]

# turns dict state into flat state
def flatten_state(state):
    flattened_state = []
    for d in state:
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
        
        # 1 if reward1 >= reward2, -1 otherwise
        feature_pref = 1
        if reward2 > reward1:
            feature_pref = -1
        feature_prefs.append(feature_pref)
    return feature_prefs

# finds the highest reward mushroom(s)
def calculate_best_mushroom(features, true_reward):
    best_mushrooms = []
    max_rew = 0
    # generates all possible mushrooms
    mushrooms = [{"texture": state[0], "color": state[1], "shape": state[2], "height": state[3], "weight": state[4], "smell": state[5]} 
           for state in itertools.product(features['texture'],features['color'],features['shape'],features['height'],features['weight'],features['smell'])]
    
    for mushroom in mushrooms:
        rew = calculate_reward(mushroom, true_reward)
        if rew == max_rew:
            best_mushrooms.append(flatten_state([mushroom]))
        elif rew > max_rew:
            best_mushrooms.clear()
            best_mushrooms.append(flatten_state([mushroom]))
            max_rew = rew
    return best_mushrooms, max_rew