import argparse
import matplotlib.pyplot as plt

from feature_preference.utils.mushroom_utils import calc_num_labels, plot_comparisons, plot_labels

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='flights')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
########################################################################

# parses seed results
in_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_rlhf.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
rlhf_probs = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        rlhf_probs.append(float(line_data.split(":  ")[1]))

# parses featureprefs results
in_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_featureprefs.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
featureprefs_probs = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        featureprefs_probs.append(float(line_data.split(":  ")[1]))

# parses featureprefs_human results
in_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_featureprefshuman.txt'
with open(in_file, 'r') as f:
    file_data= f.read()

splitted_data = file_data.split("\n")

comparisons = []
featureprefshuman_probs = []

for i in range(len(splitted_data)):
    line_data = splitted_data[i]
    if "Evaluating" in line_data:
        comparisons.append(line_data.split(' ')[1])
    elif "Average probability" in line_data:
        featureprefshuman_probs.append(float(line_data.split(":  ")[1]))

# saves as processed file
out_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_parsed.txt'
with open(out_file, 'w') as f:
    f.write("comparisons = {}\n".format(comparisons))
    f.write("rlhf_probs = {}\n".format(rlhf_probs))
    f.write("featureprefs_probs = {}\n".format(featureprefs_probs))
    f.write("featureprefshuman_probs = {}\n".format(featureprefshuman_probs))

# plots
save_loc = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed)
plot_comparisons(comparisons, rlhf_probs, featureprefs_probs, featureprefshuman_probs, 'prob_gt_reward', save_loc)