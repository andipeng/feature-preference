import random
import numpy as np

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