import itertools
import random

# calculates state reward based on ground truth rewards
def calculate_reward(state, true_reward):
    reward = 0
    for i in range(len(state)):
        reward += state[i] * true_reward[i]
    return reward

def calculate_feature_prefs(state1, state2, true_reward):
    feature_prefs = []
    for i in range(len(state1)):
        # 1 if reward1 >= reward2, -1 otherwise
        feature_pref = 1
        if state2[i]*true_reward[i] > state1[i]*true_reward[i]:
            feature_pref = -1
        feature_prefs.append(feature_pref)
    return feature_prefs

# finds the highest reward flight(s)
def calculate_best_flight(true_reward):
    arrival_time = possible_feature_values(true_reward[0], 'continuous')
    american = possible_feature_values(true_reward[1], 'categorical')
    delta = possible_feature_values(true_reward[2], 'categorical')
    jetblue = possible_feature_values(true_reward[3], 'categorical')
    southwest = possible_feature_values(true_reward[4], 'categorical')
    longest_stop = possible_feature_values(true_reward[5], 'continuous')
    num_stops = possible_feature_values(true_reward[6], 'continuous')
    price = possible_feature_values(true_reward[7], 'continuous')

    best_flights = list(itertools.product(arrival_time, american, delta, jetblue, southwest, longest_stop, num_stops, price))
        
    return best_flights

def sample_state(features):
    state = []
    for feature in features:
        if len(features[feature]) == 1: # continuous sampled between 0 and specified range
            sampled_feature = round(random.uniform(0, features[feature][0]), 2)
        else: # otherwise samples from discrete values
            sampled_feature = random.sample(features[feature], 1)[0]
        state.append(sampled_feature)
    return state

# returns possible values for each feature according to reward
def possible_feature_values(reward_weight, feature_type):
    if reward_weight > 0:
        return [1.0]
    elif reward_weight < 0:
        return [0.0]
    elif reward_weight == 0:
        if feature_type == 'continuous':
            return [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            return [0.0, 1.0]

def write_flight(state1, state1_reward, state2, state2_reward, pref, feature_prefs, feature_map):
    final_list = []
    final_list.extend(state1)
    final_list.extend([state1_reward])
    final_list.extend(state2)
    final_list.extend([state2_reward])
    final_list.extend([pref])
    final_list.extend(feature_prefs)
    final_list.extend(feature_map)
    return final_list