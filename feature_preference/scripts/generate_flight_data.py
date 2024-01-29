import yaml
import argparse
import copy
import csv
import json
import random

from feature_preference.utils.flight_utils import calculate_reward, calculate_feature_prefs, write_flight, sample_state

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='flights')
parser.add_argument('--config', type=str, default='reward1')
parser.add_argument('--num_comparisons', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--test', type=bool, default=False) # generate test set

args = parser.parse_args()

# read in reward params
in_path = '../configs/' + args.env + '/' + args.config
with open(in_path + '.yaml', "r") as file:
    config = yaml.safe_load(file)
data = []
with open(in_path + '.json', 'r') as file:
    for line in file:
        data.append(json.loads(line))
########################################################################

print("========================================")
print("Sampling %d comparisons" % args.num_comparisons)
print("========================================")

if args.test:
    path_out = '../data/' + args.env + '/' + args.config + '/test_' + str(args.num_comparisons) + '.csv'
else:
    path_out = '../data/' + args.env + '/' + args.config + '/' + str(args.seed) + '/train_' + str(args.num_comparisons) + '.csv'
    path_out_human_augment = '../data/' + args.env + '/' + args.config + '/' + str(args.seed) + '/train_' + str(args.num_comparisons) + '_human_augment.csv'
    path_out_gt_augment = '../data/' + args.env + '/' + args.config + '/' + str(args.seed) + '/train_' + str(args.num_comparisons) + '_gt_augment.csv'

# train data generated from Lin et al. dataset
if not args.test:
    with open(path_out, 'w', newline='') as file1, open(path_out_human_augment, 'w', newline='') as file2, open(path_out_gt_augment, 'w', newline='') as file3:
        csv_writer = csv.writer(file1)
        csv_writer_human_augment = csv.writer(file2)
        csv_writer_gt_augment = csv.writer(file3)

        # each datapoint generates 2 comparisons
        flight_pairs = []
        for i in data:
            # gets best flight, compares to other two
            optimal_flight = i['options'][i['optimal_index']].copy()
            i['options'].remove(optimal_flight)
            flight2 = i['options'][0].copy()
            flight3 = i['options'][1].copy()

            optimal_reward = calculate_reward(optimal_flight, config['true_reward'])
            flight2_reward = calculate_reward(flight2, config['true_reward'])
            flight3_reward = calculate_reward(flight3, config['true_reward'])

            feature_prefs1 = calculate_feature_prefs(optimal_flight, flight2, config['true_reward'])
            feature_prefs2 = calculate_feature_prefs(flight3, optimal_flight, config['true_reward'])

            # [state1 (8), state1reward (1), state2 (8), state2reward (1), pref (1), feature_prefs (8), human_feature_map (8), gt_feature_map (8)]
            flight_pairs.append(write_flight(optimal_flight, optimal_reward, flight2, flight2_reward, 1, feature_prefs1, config['human_feature_map'], config['gt_feature_map']))
            flight_pairs.append(write_flight(flight3, flight3_reward, optimal_flight, optimal_reward, -1, feature_prefs2, config['human_feature_map'], config['gt_feature_map']))
        
        # randomly selects pairs from the full set
        for _ in range(args.num_comparisons):
            random_pair = random.sample(flight_pairs, 1)[0]
            csv_writer.writerow(random_pair)
            csv_writer_human_augment.writerow(random_pair)
            csv_writer_gt_augment.writerow(random_pair)
        
            # performs augmentation of non-relevant features
            for i in range(len(config['human_feature_map'])):
                if config['human_feature_map'][i] == 0:
                    state1_copy = random_pair[0:8].copy()
                    state2_copy = random_pair[9:17].copy()
                    state1_copy[i] = copy.copy(random_pair[9:17][i])
                    state2_copy[i] = copy.copy(random_pair[0:8][i])
                    state = write_flight(state1_copy, random_pair[8], state2_copy, random_pair[17], random_pair[18], random_pair[19:27], config['human_feature_map'], config['gt_feature_map'])
                    csv_writer_human_augment.writerow(state)
        
            for i in range(len(config['gt_feature_map'])):
                if config['gt_feature_map'][i] == 0:
                    state1_copy = random_pair[0:8].copy()
                    state2_copy = random_pair[9:17].copy()
                    state1_copy[i] = copy.copy(random_pair[9:17][i])
                    state2_copy[i] = copy.copy(random_pair[0:8][i])
                    state = write_flight(state1_copy, random_pair[8], state2_copy, random_pair[17], random_pair[18], random_pair[19:27], config['human_feature_map'], config['gt_feature_map'])
                    csv_writer_gt_augment.writerow(state)

# test data generated from reward weights
else:
    with open(path_out, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        for _ in range(args.num_comparisons):
            # samples 2 states and obtains rewards (state only)
            state1 = sample_state(config['features'])
            state2 = sample_state(config['features'])
            state1_reward = calculate_reward(state1, config['true_reward'])
            state2_reward = calculate_reward(state2, config['true_reward'])

            # calculates per feature preferences
            feature_prefs = calculate_feature_prefs(state1, state2, config['true_reward'])

            # saves preferences (1 if reward1 >= reward2, -1 otherwise)
            pref = 1
            if state2_reward > state1_reward:
                pref = -1

            # [state1 (8), state1reward (1), state2 (8), state2reward (1), pref (1), feature_prefs (8), human_feature_map (8)]
            state = write_flight(state1, state1_reward, state2, state2_reward, pref, feature_prefs, config['human_feature_map'], config['gt_feature_map'])
            csv_writer.writerow(state)