import argparse
import matplotlib.pyplot as plt
import numpy as np

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
########################################################################

# plotting code
def plot(x, y1, y2, y3, y_label, save_loc):
    # create an index list for x-values
    x_values = range(len(x))

    fig, ax = plt.subplots()
    ax.plot(x_values, y1, marker='o', color='black', label='rlhf')
    ax.plot(x_values, y2, marker='o', color='green', label='feature_prefs')
    ax.plot(x_values, y3, marker='o', color='orange', label='feature_prefs_human')

    # set x-ticks to be the comparison values
    ax.set_xticks(x_values)
    ax.set_xticklabels(x)
    ax.yaxis.set_ticks(np.arange(0.5, 1.05, 0.1))

    ax.set_xlabel('Number of Comparisons')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(save_loc + '/0' + y_label + '.pdf')

# parses seed results
in_file = '../results/sim_mushrooms/' + args.reward + '/' + str(args.seed) + '/0results_rlhf.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
rlhf_probs = []
rlhf_correct = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        rlhf_probs.append(float(line_data.split(":  ")[1]))
    elif "Percent correct" in line_data:
        rlhf_correct.append(float(line_data.split(": ")[1]))

# parses featureprefs results
in_file = '../results/sim_mushrooms/' + args.reward + '/' + str(args.seed) + '/0results_featureprefs.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
featureprefs_probs = []
featureprefs_correct = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        featureprefs_probs.append(float(line_data.split(":  ")[1]))
    elif "Percent correct" in line_data:
        featureprefs_correct.append(float(line_data.split(": ")[1]))

# parses featureprefs_human results
in_file = '../results/sim_mushrooms/' + args.reward + '/' + str(args.seed) + '/0results_featureprefshuman.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
featureprefshuman_probs = []
featureprefshuman_correct = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        featureprefshuman_probs.append(float(line_data.split(":  ")[1]))
    elif "Percent correct" in line_data:
        featureprefshuman_correct.append(float(line_data.split(": ")[1]))

# saves as processed file
out_file = '../results/sim_mushrooms/' + args.reward + '/' + str(args.seed) + '/0results_parsed.txt'
with open(out_file, 'w') as f:
    f.write("comparisons = {}\n".format(comparisons))
    f.write("rlhf_probs = {}\n".format(rlhf_probs))
    f.write("rlhf_correct = {}\n".format(rlhf_correct))
    f.write("featureprefs_probs = {}\n".format(featureprefs_probs))
    f.write("featureprefs_correct = {}\n".format(featureprefs_correct))
    f.write("featureprefshuman_probs = {}\n".format(featureprefshuman_probs))
    f.write("featureprefshuman_correct = {}\n".format(featureprefshuman_correct))

# plots
save_loc = '../results/sim_mushrooms/' + args.reward + '/' + str(args.seed)
plot(comparisons, rlhf_probs, featureprefs_probs, featureprefshuman_probs, 'prob_gt_reward', save_loc)
plot(comparisons, rlhf_correct, featureprefs_correct, featureprefshuman_correct, 'accuracy_test_set', save_loc)