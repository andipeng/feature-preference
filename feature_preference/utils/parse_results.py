import argparse
import matplotlib.pyplot as plt
import ast
import statistics
import math

from feature_preference.utils.mushroom_utils import calc_num_labels, plot_comparisons, plot_labels

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--seeds', type=list, default=[1,2,3])
parser.add_argument('--rel_features', type=int, default=3)

args = parser.parse_args()
########################################################################

def standard_error(values):
    return statistics.stdev(values) / math.sqrt(len(values))

def calc_avg(*lists):
    averages = [sum(values) / len(values) for values in zip(*lists)]
    standard_errors = [standard_error(values) for values in zip(*lists)]
    return averages, standard_errors

# Function to parse file and return the lists
def parse_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    lines = content.split('\n')
    data = {}
    for line in lines:
        if line:
            key, value = line.split(' = ')
            data[key] = ast.literal_eval(value)
    return data

# Assuming the files are named file1.py, file2.py, file3.py
files = []
for seed in args.seeds:
    filename = '../results/'  + args.env + '/' + args.reward + '/' + str(seed) + '/0results_parsed.txt'
    files.append(filename)

# Parse each file and get the average of the lists
comparisons = parse_file(filename)['comparisons']
rlhf_probs, rlhf_probs_err = calc_avg(*[parse_file(file)['rlhf_probs'] for file in files])
rlhf_correct, rlhf_cor_err = calc_avg(*[parse_file(file)['rlhf_correct'] for file in files])

featureprefs_probs, featureprefs_probs_err = calc_avg(*[parse_file(file)['featureprefs_probs'] for file in files])
featureprefs_correct, featureprefs_cor_err = calc_avg(*[parse_file(file)['featureprefs_correct'] for file in files])

featureprefshuman_probs, featureprefshuman_probs_err = calc_avg(*[parse_file(file)['featureprefshuman_probs'] for file in files])
featureprefshuman_correct, featureprefshuman_cor_err = calc_avg(*[parse_file(file)['featureprefshuman_correct'] for file in files])

rlhfhuman_probs, rlhfhuman_probs_err = calc_avg(*[parse_file(file)['rlhfhuman_probs'] for file in files])
rlhfhuman_correct, rlhfhuman_cor_err = calc_avg(*[parse_file(file)['rlhfhuman_correct'] for file in files])

# saves as processed file
out_file = '../results/'  + args.env + '/' + args.reward + '/results.txt'
with open(out_file, 'w') as f:
    f.write("comparisons = {}\n".format(comparisons))
    f.write("rlhf_probs = {}\n".format(rlhf_probs))
    f.write("rlhf_probs_err = {}\n".format(rlhf_probs_err))
    f.write("rlhf_correct = {}\n".format(rlhf_correct))
    f.write("rlhf_cor_err = {}\n".format(rlhf_cor_err))
    f.write("featureprefs_probs = {}\n".format(featureprefs_probs))
    f.write("featureprefs_probs_err = {}\n".format(featureprefs_probs_err))
    f.write("featureprefs_correct = {}\n".format(featureprefs_correct))
    f.write("featureprefs_cor_err = {}\n".format(featureprefs_cor_err))
    f.write("featureprefshuman_probs = {}\n".format(featureprefshuman_probs))
    f.write("featureprefshuman_probs_err = {}\n".format(featureprefshuman_probs_err))
    f.write("featureprefshuman_correct = {}\n".format(featureprefshuman_correct))
    f.write("featureprefshuman_cor_err = {}\n".format(featureprefshuman_cor_err))
    f.write("rlhfhuman_probs = {}\n".format(rlhfhuman_probs))
    f.write("rlhfhuman_probs_err = {}\n".format(rlhfhuman_probs_err))
    f.write("rlhfhuman_correct = {}\n".format(rlhfhuman_correct))
    f.write("rlhfhuman_cor_err = {}\n".format(rlhfhuman_cor_err))

if args.env == 'sim_mushrooms':
    rlhf_labels = [1,3,5,10,15,20,30,50,100]
elif args.env == 'flights':
    rlhf_labels = [1,3,5,10]
featureprefs_labels = calc_num_labels(rlhf_labels, 6) # calculates all feature labels
featureprefshuman_labels = calc_num_labels(rlhf_labels, args.rel_features) # calculates only human specified ones

# plots
save_loc = '../results/'  + args.env + '/' + args.reward
plot_comparisons(comparisons, rlhf_probs, featureprefs_probs, featureprefshuman_probs, rlhfhuman_probs, 'prob_gt_reward', save_loc, rlhf_probs_err, featureprefs_probs_err, featureprefshuman_probs_err, rlhfhuman_probs_err)
plot_comparisons(comparisons, rlhf_correct, featureprefs_correct, featureprefshuman_correct, rlhfhuman_correct, 'accuracy_test_set', save_loc, rlhf_cor_err, featureprefs_cor_err, featureprefshuman_cor_err, rlhfhuman_cor_err)

plot_labels(rlhf_labels, featureprefs_labels, featureprefshuman_labels, rlhf_labels, rlhf_probs, featureprefs_probs, featureprefshuman_probs, rlhfhuman_probs, 'prob_gt_reward', save_loc, rlhf_probs_err, featureprefs_probs_err, featureprefshuman_probs_err, rlhfhuman_probs_err)
plot_labels(rlhf_labels, featureprefs_labels, featureprefshuman_labels, rlhf_labels, rlhf_correct, featureprefs_correct, featureprefshuman_correct, rlhfhuman_correct, 'accuracy_test_set', save_loc, rlhf_cor_err, featureprefs_cor_err, featureprefshuman_cor_err, rlhfhuman_cor_err)
