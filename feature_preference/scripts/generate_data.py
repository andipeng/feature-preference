import yaml
import argparse
import csv

from feature_preference.utils.mushroom_utils import sample_state, flatten_state, calculate_feature_prefs, calculate_reward

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--config', type=str, default='reward1')
parser.add_argument('--num_comparisons', type=int, default=5)
parser.add_argument('--test', type=bool, default=False)

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
# samples random mushroom from all possible features

with open(path_out, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    for _ in range(args.num_comparisons):
        final_list = []

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

        # [state1, state1reward, state2, state2reward, pref, feature_prefs]
        final_list.extend(state1)
        final_list.extend([state1_reward])
        final_list.extend(state2)
        final_list.extend([state2_reward])
        final_list.extend([pref])
        final_list.extend(feature_prefs)
        csv_writer.writerow(final_list)
