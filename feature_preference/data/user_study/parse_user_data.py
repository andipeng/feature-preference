import csv
import argparse
import yaml
import random

from feature_preference.utils.mushroom_utils import flatten_state, calculate_user_feature_prefs, calculate_user_reward, write_mushroom

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='user_study')
parser.add_argument('--config', type=str, default='reward1')
parser.add_argument('--num_comparisons', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

# read in reward params
yaml_path = '../../configs/' + args.env + '/' + args.config + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################
    
# reads in user data file (replace with path to actual data)
with open('../user_study/mushroom_features_groundtruthRewards.csv', 'r', newline='') as gt_file:
    csv_reader = csv.reader(gt_file)
    
    user_data = []
    for row in csv_reader:
        user_data.append([int(x) for x in row])

# extracts corresponding reward data from user responses
reward_num = int(args.config[-1])
user_data = user_data[reward_num - 1:reward_num + 5 - 1]

path_out = '../../data/' + args.env + '/' + args.config + '/' + str(args.seed) + '/train_' + str(args.num_comparisons) + '.csv'
path_out_augment = '../../data/' + args.env + '/' + args.config + '/' + str(args.seed) + '/train_' + str(args.num_comparisons) + '_augment.csv'

with open(path_out, 'w', newline='') as file1, open(path_out_augment, 'w', newline='') as file2:
    csv_writer = csv.writer(file1)
    csv_writer_augment = csv.writer(file2)
    
    for _ in range(args.num_comparisons):
        # randomly samples from user data
        comparison = random.sample(user_data, 1)[0]
        # samples 2 states and obtains rewards (state only)
        state1 = comparison[0:18]
        state2 = comparison[18:36]
        state1_reward = calculate_user_reward(state1, config['true_reward'])
        state2_reward = calculate_user_reward(state2, config['true_reward'])

        # calculates per feature preferences
        feature_prefs = calculate_user_feature_prefs(state1, state2, config['true_reward'])

        # saves preferences (1 if reward1 >= reward2, -1 otherwise)
        pref = 1
        if state2_reward > state1_reward:
            pref = -1

        # [state1 (18), state1reward (1), state2 (18), state2reward (1), pref (1), feature_prefs (6), human_feature_map (6)]
        state = write_mushroom(state1, state1_reward, state2, state2_reward, pref, feature_prefs, config['human_feature_map'])
        csv_writer.writerow(state)
        csv_writer_augment.writerow(state)

        # performs augmentation of non-relevant features
        for i in range(len(config['human_feature_map'])):
            if config['human_feature_map'][i] == 0:
                state1_copy = state1.copy()
                state2_copy = state2.copy()
                state1_copy[i*3:i*3+3] = state2[i*3:i*3+3].copy()
                state2_copy[i*3:i*3+3] = state1[i*3:i*3+3].copy()
                state = write_mushroom(state1_copy, state1_reward, state2_copy, state2_reward, pref, feature_prefs, config['human_feature_map'])
                csv_writer_augment.writerow(state)

