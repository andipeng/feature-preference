import argparse
import matplotlib.pyplot as plt

from feature_preference.utils.mushroom_utils import calc_num_labels, plot_mushroom_comparisons, plot_labels
from feature_preference.utils.flight_utils import plot_flight_comparisons

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward3')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--rel_features', type=int, default=1)

args = parser.parse_args()
########################################################################

# parses rlhf results
in_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_rlhf.txt'
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
if args.env == 'sim_mushrooms':
    in_file = '../results/'  + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_featureprefs.txt'
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
in_file = '../results/'  + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_featureprefshuman.txt'
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

# parses rlhf_human results
if args.env == 'sim_mushrooms':
    in_file = '../results/'  + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_rlhfhuman.txt'
    with open(in_file, 'r') as f:
        file_data= f.read()

    splitted_data = file_data.split("\n")

    comparisons = []
    rlhfhuman_probs = []
    rlhfhuman_correct = []

    for i in range(len(splitted_data)):
        line_data = splitted_data[i]
        if "Evaluating" in line_data:
            comparisons.append(line_data.split(' ')[1])
        elif "Average probability" in line_data:
            rlhfhuman_probs.append(float(line_data.split(":  ")[1]))
        elif "Percent correct" in line_data:
            rlhfhuman_correct.append(float(line_data.split(": ")[1]))

# saves as processed file
out_file = '../results/'  + args.env + '/' + args.reward + '/' + str(args.seed) + '/0results_parsed.txt'
with open(out_file, 'w') as f:
    f.write("comparisons = {}\n".format(comparisons))
    f.write("rlhf_probs = {}\n".format(rlhf_probs))
    f.write("rlhf_correct = {}\n".format(rlhf_correct))
    if args.env == 'sim_mushrooms':
        f.write("featureprefs_probs = {}\n".format(featureprefs_probs))
        f.write("featureprefs_correct = {}\n".format(featureprefs_correct))
    f.write("featureprefshuman_probs = {}\n".format(featureprefshuman_probs))
    f.write("featureprefshuman_correct = {}\n".format(featureprefshuman_correct))
    if args.env == 'sim_mushrooms':
        f.write("rlhfhuman_probs = {}\n".format(rlhfhuman_probs))
        f.write("rlhfhuman_correct = {}\n".format(rlhfhuman_correct))

#if args.env == 'sim_mushrooms':
#    rlhf_labels = [1,3,5,10,15,20,30,50,100]
#elif args.env == 'flights':
#    rlhf_labels = [1,3,5,10]
#featureprefs_labels = calc_num_labels(rlhf_labels, 6) # calculates all feature labels
#featureprefshuman_labels = calc_num_labels(rlhf_labels, args.rel_features) # calculates only human specified ones

# plots
save_loc = '../results/'  + args.env + '/' + args.reward + '/' + str(args.seed)
if args.env == 'sim_mushrooms':
    plot_mushroom_comparisons(comparisons, rlhf_probs, featureprefs_probs, featureprefshuman_probs, rlhfhuman_probs, 'prob_gt_reward', save_loc)
    plot_mushroom_comparisons(comparisons, rlhf_correct, featureprefs_correct, featureprefshuman_correct, rlhfhuman_correct, 'accuracy_test_set', save_loc)
elif args.env == 'flights':
    plot_flight_comparisons(comparisons, rlhf_probs, featureprefshuman_probs, 'prob_gt_reward', save_loc)
    plot_flight_comparisons(comparisons, rlhf_correct, featureprefshuman_correct, 'accuracy_test_set', save_loc)

# plot_labels(rlhf_labels, featureprefs_labels, featureprefshuman_labels, rlhf_labels, rlhf_probs, featureprefs_probs, featureprefshuman_probs, rlhfhuman_probs, 'prob_gt_reward', save_loc)
# plot_labels(rlhf_labels, featureprefs_labels, featureprefshuman_labels, rlhf_labels, rlhf_correct, featureprefs_correct, featureprefshuman_correct, rlhfhuman_correct, 'accuracy_test_set', save_loc)