import argparse
import matplotlib.pyplot as plt
import ast
import statistics
import math
import numpy as np

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--seeds', type=list, default=[1,2,3])

args = parser.parse_args()
########################################################################

# plotting code
def plot(x, y1, y1_err, y2, y2_err, y3, y3_err, y_label, save_loc):
    # create an index list for x-values
    x_values = range(len(x))
    y1, y1_err = np.array(y1), np.array(y1_err)
    y2, y2_err = np.array(y2), np.array(y2_err)
    y3, y3_err = np.array(y3), np.array(y3_err)

    fig, ax = plt.subplots()
    ax.fill_between(x_values, y1-y1_err, y1+y1_err, color='black', alpha=0.1)
    ax.fill_between(x_values, y2-y2_err, y2+y2_err, color='green', alpha=0.1)
    ax.fill_between(x_values, y3-y3_err, y3+y3_err, color='orange', alpha=0.1)

    ax.plot(x_values, y1, marker='o', color='black', label='rlhf')
    ax.plot(x_values, y2, marker='o', color='green', label='feature_prefs')
    ax.plot(x_values, y3, marker='o', color='orange', label='feature_prefs_human')

    # set x-ticks to be the comparison values
    ax.set_xticks(x_values)
    ax.set_xticklabels(x)
    ax.yaxis.set_ticks(np.arange(0, 1, 0.2))

    ax.set_xlabel('Number of Comparisons')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(save_loc + '/' + y_label + '.pdf')

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
    filename = '../results/sim_mushrooms/' + args.reward + '/' + str(seed) + '/0results_parsed.txt'
    files.append(filename)

# Parse each file and get the average of the lists
comparisons = parse_file(filename)['comparisons']
rlhf_probs, rlhf_probs_err = calc_avg(*[parse_file(file)['rlhf_probs'] for file in files])
rlhf_correct, rlhf_cor_err = calc_avg(*[parse_file(file)['rlhf_correct'] for file in files])

featureprefs_probs, featureprefs_probs_err = calc_avg(*[parse_file(file)['featureprefs_probs'] for file in files])
featureprefs_correct, featureprefs_cor_err = calc_avg(*[parse_file(file)['featureprefs_correct'] for file in files])

featureprefshuman_probs, featureprefshuman_probs_err = calc_avg(*[parse_file(file)['featureprefshuman_probs'] for file in files])
featureprefshuman_correct, featureprefshuman_cor_error = calc_avg(*[parse_file(file)['featureprefshuman_correct'] for file in files])

# saves as processed file
out_file = '../results/sim_mushrooms/' + args.reward + '/results.txt'
with open(out_file, 'w') as f:
    f.write("comparisons = {}\n".format(comparisons))
    f.write("rlhf_probs = {}\n".format(rlhf_probs))
    f.write("rlhf_correct = {}\n".format(rlhf_correct))
    f.write("featureprefs_probs = {}\n".format(featureprefs_probs))
    f.write("featureprefs_correct = {}\n".format(featureprefs_correct))
    f.write("featureprefshuman_probs = {}\n".format(featureprefshuman_probs))
    f.write("featureprefshuman_correct = {}\n".format(featureprefshuman_correct))

# plots
save_loc = '../results/sim_mushrooms/' + args.reward
plot(comparisons, rlhf_probs, rlhf_probs_err, featureprefs_probs, featureprefs_probs_err, featureprefshuman_probs, featureprefshuman_probs_err, 'prob_gt_reward', save_loc)
plot(comparisons, rlhf_correct, rlhf_cor_err, featureprefs_correct, featureprefs_cor_err, featureprefshuman_correct, featureprefshuman_cor_error, 'accuracy_test_set', save_loc)