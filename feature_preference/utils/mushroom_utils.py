import random
import numpy as np
import itertools
import matplotlib.pyplot as plt

# plotting code
def plot_mushroom_comparisons(x, y1, y2, y3, y4, y_label, save_loc, y1_err=None, y2_err=None, y3_err=None, y4_err=None):
    # create an index list for x-values
    x_values = range(len(x))
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)

    fig, ax = plt.subplots()
    # add std err if over multiple seeds
    if y1_err is not None:
        y1_err = np.array(y1_err)
        y2_err = np.array(y2_err)
        y3_err = np.array(y3_err)
        y4_err = np.array(y4_err)
        ax.fill_between(x_values, y1-y1_err, y1+y1_err, color='black', alpha=0.1)
        ax.fill_between(x_values, y2-y2_err, y2+y2_err, color='green', alpha=0.1)
        ax.fill_between(x_values, y3-y3_err, y3+y3_err, color='deeppink', alpha=0.1)
        ax.fill_between(x_values, y4-y4_err, y4+y4_err, color='blue', alpha=0.1)
    ax.plot(x_values, y1, marker='o', color='black', label='rlhf')
    ax.plot(x_values, y2, marker='o', color='green', label='feature_prefs')
    ax.plot(x_values, y3, marker='o', color='deeppink', label='feature_prefs_human')
    ax.plot(x_values, y4, marker='o', color='blue', label='rlhf_human')

    # set x-ticks to be the comparison values
    ax.set_xticks(x_values)
    ax.set_xticklabels(x)
    ax.yaxis.set_ticks(np.arange(0.5, 1.05, 0.1))

    ax.set_xlabel('Number of Comparisons')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(save_loc + '/0' + y_label + '_comparisons.pdf')

def plot_labels(x1, x2, x3, x4, y1, y2, y3, y4, y_label, save_loc, y1_err=None, y2_err=None, y3_err=None, y4_err=None):
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)

    fig, ax = plt.subplots()
    if y1_err is not None:
        y1_err = np.array(y1_err)
        y2_err = np.array(y2_err)
        y3_err = np.array(y3_err)
        y4_err = np.array(y4_err)
        ax.fill_between(x1, y1-y1_err, y1+y1_err, color='black', alpha=0.1)
        ax.fill_between(x2, y2-y2_err, y2+y2_err, color='green', alpha=0.1)
        ax.fill_between(x3, y3-y3_err, y3+y3_err, color='orange', alpha=0.1)
        ax.fill_between(x4, y4-y4_err, y4+y4_err, color='blue', alpha=0.1)

    ax.plot(x1, y1, marker='o', color='black', label='rlhf')
    ax.plot(x2, y2, marker='o', color='green', label='feature_prefs')
    ax.plot(x3, y3, marker='o', color='orange', label='feature_prefs_human')
    ax.plot(x4, y4, marker='o', color='blue', label='rlhf_human')

    ax.yaxis.set_ticks(np.arange(0.5, 1.05, 0.1))
    ax.set_xlim([0, 50])

    ax.set_xlabel('Number of Labels')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(save_loc + '/0' + y_label + '_labels.pdf')

def calc_num_labels(list, features):
    feature_labels = [i * features for i in list]
    tot_labels = [x + y for x, y in zip(list, feature_labels)]
    return tot_labels

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

def write_mushroom(state1, state1_reward, state2, state2_reward, pref, feature_prefs, feature_map):
    final_list = []
    final_list.extend(state1)
    final_list.extend([state1_reward])
    final_list.extend(state2)
    final_list.extend([state2_reward])
    final_list.extend([pref])
    final_list.extend(feature_prefs)
    final_list.extend(feature_map)
    return final_list