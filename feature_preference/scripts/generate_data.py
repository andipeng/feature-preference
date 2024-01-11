import yaml
import argparse
import csv

from feature_preference.utils.mushroom_utils import sample_state, flatten_state, calculate_feature_prefs, calculate_reward, write_mushroom

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--config', type=str, default='reward5')
parser.add_argument('--num_comparisons', type=int, default=1)
parser.add_argument('--test', type=bool, default=False) # generate test set
parser.add_argument('--augment', type=bool, default=True) # augment with non-relevant feature swapping

args = parser.parse_args()

# read in reward params
yaml_path = '../configs/' + args.env + '/' + args.config + '.yaml'
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)
########################################################################

print("========================================")
print("Sampling %d comparisons" % args.num_comparisons)
print("========================================")

if args.test:
    path_out = '../data/' + args.env + '/' + args.config + '/test_' + str(args.num_comparisons) + '.csv'
else:
    path_out = '../data/' + args.env + '/' + args.config + '/train_' + str(args.num_comparisons) + '.csv'
    path_out_augment = '../data/' + args.env + '/' + args.config + '/train_' + str(args.num_comparisons) + '_augment.csv'
# samples random mushroom from all possible features

with open(path_out, 'w', newline='') as file1, open(path_out_augment, 'w', newline='') as file2:
    csv_writer = csv.writer(file1)
    csv_writer_augment = csv.writer(file2)
    
    for _ in range(args.num_comparisons):
        # samples 2 states and obtains rewards (state only)
        state1 = sample_state(config['features'])
        state2 = sample_state(config['features'])
        state1_reward = calculate_reward(state1[0], config['true_reward'])
        state2_reward = calculate_reward(state2[0], config['true_reward'])

        # calculates per feature preferences
        feature_prefs = calculate_feature_prefs(state1[0], state2[0], config['true_reward'])

        state1 = flatten_state(state1)
        state2 = flatten_state(state2)

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