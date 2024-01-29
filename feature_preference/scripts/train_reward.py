import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim

from feature_preference.models.reward_networks import LinearRewardMLP, PairwiseLoss, FeaturePrefNetworkMushrooms, FeaturePrefNetworkFlights

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='rlhf_human') # rlhf, feature_prefs (joint loss), feature_prefs_human (pragmatic joint loss), rlhf_human (pragmatic rlhf), feature_prefs_gt (pragmatic joint loss, gt feature map)
parser.add_argument('--linear', type=bool, default=False)
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--data_file', type=str, default='train_1')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.5) # param for feature_weight (feature_prefs)
parser.add_argument('--beta', type=float, default=0.5) # param for state_weight (feature_prefs)

args = parser.parse_args()
########################################################################

if args.env == 'sim_mushrooms':
    if args.prefs_type == 'feature_prefs_human' or args.prefs_type == 'rlhf_human':
        data_file = '../data/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.data_file + '_augment.csv'
    else:
        data_file = '../data/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.data_file + '.csv'
elif args.env == 'flights':
    if args.prefs_type == 'feature_prefs_human' or args.prefs_type == 'rlhf_human':
        data_file = '../data/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.data_file + '_human_augment.csv'
    elif args.prefs_type == 'feature_prefs_gt':
        data_file = '../data/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.data_file + '_gt_augment.csv'
    else:
        data_file = '../data/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.data_file + '.csv'
with open(data_file) as file_obj:
    reader_obj = csv.reader(file_obj)

    states1 = []
    states2 = []
    prefs = []
    feature_prefs = []
    feature_maps = []
    for row in reader_obj:
        if args.env == 'sim_mushrooms':
            states1.append(row[0:18])
            states2.append(row[19:37])
            prefs.append([row[38]])
            feature_prefs.append(row[39:45])
            feature_maps.append([1,1,1,1,1,1])
        elif args.env == 'flights':
            states1.append(row[0:8])
            states2.append(row[9:17])
            prefs.append([row[18]])
            feature_prefs.append(row[19:27])
            feature_maps.append([1,1,1,1,1,1,1,1])
    states1 = np.array(states1,dtype=float)
    states2 = np.array(states2,dtype=float)
    prefs = np.array(prefs,dtype=float)
    feature_prefs = np.array(feature_prefs,dtype=float)
    feature_maps = np.array(feature_maps,dtype=float)

print("========================================")
print("Loaded data from " + args.data_file)
print("========================================")

# defines network depending on type of comparison(s)
if args.prefs_type == 'rlhf' or args.prefs_type == 'rlhf_human':
    reward_net = LinearRewardMLP(state_dim=len(states1[0]))
else:
    if args.env == 'sim_mushrooms':
        reward_net = FeaturePrefNetworkMushrooms(feature_dim=3, num_features=6)
    elif args.env == 'flights':
        reward_net = FeaturePrefNetworkFlights(feature_dim=1, num_features=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_net.to(device)

loss_fn = PairwiseLoss()
optimizer = optim.Adam(list(reward_net.parameters()), lr=0.001)

losses = []

idxs = np.array(range(len(states1)))
args.num_batches = len(idxs) // args.batch_size

# Train the model with regular SGD
for epoch in range(args.epochs):
    running_loss = 0.0
    for i in range(args.num_batches):
        optimizer.zero_grad()
   
        states1 = torch.Tensor(states1).float().to(device)
        states2 = torch.Tensor(states2).float().to(device)
        prefs = torch.Tensor(prefs).float().to(device)
        feature_prefs = torch.Tensor(feature_prefs).float().to(device)
        feature_maps = torch.Tensor(feature_maps).float().to(device)

        # pairwise ranking loss (1 -> pred1 better than pred2, -1 -> the other way around)
        if args.prefs_type == 'rlhf' or args.prefs_type == 'rlhf_human':
            preds_r1 = reward_net(states1)
            preds_r2 = reward_net(states2)
            loss = loss_fn(preds_r1, preds_r2, prefs)
        # joint loss between all features (potentially with augmented feature dataset)
        else:
            if args.env == 'sim_mushrooms':
                preds_feat1a, preds_feat2a, preds_feat3a, preds_feat4a, preds_feat5a, preds_feat6a, preds_r1 = reward_net(states1)
                preds_feat1b, preds_feat2b, preds_feat3b, preds_feat4b, preds_feat5b, preds_feat6b, preds_r2 = reward_net(states2)
                loss_feat1 = loss_fn(preds_feat1a, preds_feat1b, feature_prefs[:,0], feature_maps[:,0])
                loss_feat2 = loss_fn(preds_feat2a, preds_feat2b, feature_prefs[:,1], feature_maps[:,1])
                loss_feat3 = loss_fn(preds_feat3a, preds_feat3b, feature_prefs[:,2], feature_maps[:,2])
                loss_feat4 = loss_fn(preds_feat4a, preds_feat4b, feature_prefs[:,3], feature_maps[:,3])
                loss_feat5 = loss_fn(preds_feat5a, preds_feat5b, feature_prefs[:,4], feature_maps[:,4])
                loss_feat6 = loss_fn(preds_feat6a, preds_feat6b, feature_prefs[:,5], feature_maps[:,5])

                loss = args.alpha*(loss_feat1+loss_feat2+loss_feat3+loss_feat4+loss_feat5+loss_feat6) + args.beta*loss_fn(preds_r1, preds_r2, prefs)
            elif args.env == 'flights':
                preds_feat1a, preds_feat2a, preds_feat3a, preds_feat4a, preds_feat5a, preds_feat6a, preds_feat7a, preds_feat8a, preds_r1 = reward_net(states1)
                preds_feat1b, preds_feat2b, preds_feat3b, preds_feat4b, preds_feat5b, preds_feat6b, preds_feat7b, preds_feat8b, preds_r2 = reward_net(states2)
                loss_feat1 = loss_fn(preds_feat1a, preds_feat1b, feature_prefs[:,0], feature_maps[:,0])
                loss_feat2 = loss_fn(preds_feat2a, preds_feat2b, feature_prefs[:,1], feature_maps[:,1])
                loss_feat3 = loss_fn(preds_feat3a, preds_feat3b, feature_prefs[:,2], feature_maps[:,2])
                loss_feat4 = loss_fn(preds_feat4a, preds_feat4b, feature_prefs[:,3], feature_maps[:,3])
                loss_feat5 = loss_fn(preds_feat5a, preds_feat5b, feature_prefs[:,4], feature_maps[:,4])
                loss_feat6 = loss_fn(preds_feat6a, preds_feat6b, feature_prefs[:,5], feature_maps[:,5])
                loss_feat7 = loss_fn(preds_feat7a, preds_feat7b, feature_prefs[:,6], feature_maps[:,6])
                loss_feat8 = loss_fn(preds_feat8a, preds_feat8b, feature_prefs[:,7], feature_maps[:,7])

                loss = args.alpha*(loss_feat1+loss_feat2+loss_feat3+loss_feat4+loss_feat5+loss_feat6+loss_feat7+loss_feat8) + args.beta*loss_fn(preds_r1, preds_r2, prefs)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss))
            losses.append(running_loss)
            running_loss = 0.0
        losses.append(loss.item())

save_file = '../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.prefs_type + '_' + args.data_file + '.pt'
torch.save(reward_net, save_file)
print('Finished Training')
plt.plot(losses)
plt.savefig('../results/' + args.env + '/' + args.reward + '/' + str(args.seed) + '/' + args.prefs_type + '_' + args.data_file + '_losses.png')