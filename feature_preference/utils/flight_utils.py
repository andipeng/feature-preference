import itertools
import random
import numpy as np
import matplotlib.pyplot as plt

# plotting code
def plot_flight_comparisons(x, y1, y2, y3, y_label, save_loc, y1_err=None, y2_err=None, y3_err=None):
    # create an index list for x-values
    x_values = range(len(x))
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)

    fig, ax = plt.subplots()
    # add std err if over multiple seeds
    if y1_err is not None:
        y1_err = np.array(y1_err)
        y2_err = np.array(y2_err)
        y3_err = np.array(y3_err)
        ax.fill_between(x_values, y1-y1_err, y1+y1_err, color='black', alpha=0.1)
        ax.fill_between(x_values, y2-y2_err, y2+y2_err, color='deeppink', alpha=0.1)
        ax.fill_between(x_values, y3-y3_err, y3+y3_err, color='orange', alpha=0.1)
    ax.plot(x_values, y1, marker='o', color='black', label='rlhf')
    ax.plot(x_values, y2, marker='o', color='deeppink', label='feature_prefs_human')
    ax.plot(x_values, y3, marker='o', color='orange', label='feature_prefs_gt')

    # set x-ticks to be the comparison values
    ax.set_xticks(x_values)
    ax.set_xticklabels(x)
    ax.yaxis.set_ticks(np.arange(0.5, 1.05, 0.1))

    ax.set_xlabel('Number of Comparisons')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(save_loc + '/0' + y_label + '_comparisons.pdf')

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

def write_flight(state1, state1_reward, state2, state2_reward, pref, feature_prefs, human_feature_map, gt_feature_map):
    final_list = []
    final_list.extend(state1)
    final_list.extend([state1_reward])
    final_list.extend(state2)
    final_list.extend([state2_reward])
    final_list.extend([pref])
    final_list.extend(feature_prefs)
    final_list.extend(human_feature_map)
    final_list.extend(gt_feature_map)
    return final_list