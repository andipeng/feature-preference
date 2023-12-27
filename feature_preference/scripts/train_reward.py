import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim

from feature_preference.models.reward_networks import LinearRewardMLP

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--prefs_type', type=str, default='rlhf')
parser.add_argument('--linear', type=bool, default=True)
parser.add_argument('--env', type=str, default='sim_mushrooms')
parser.add_argument('--reward', type=str, default='reward1')
parser.add_argument('--data_file', type=str, default='train_20')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=10)

args = parser.parse_args()
########################################################################

data_file = '../data/' + args.env + '/' + args.reward + '/' + args.data_file + '.csv'
with open(data_file) as file_obj:
    reader_obj = csv.reader(file_obj)

    states1 = []
    states2 = []
    prefs = []
    for row in reader_obj:
        states1.append(row[0:18])
        states2.append(row[19:37])
        prefs.append(row[38])
    states1 = np.array(states1,dtype=int)
    states2 = np.array(states2,dtype=int)
    prefs = np.array(prefs,dtype=int)

print("========================================")
print("Loaded data from " + args.data_file)
print("========================================")

reward_net = LinearRewardMLP(state_dim=len(states1[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_net.to(device)

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
        prefs = torch.Tensor(prefs).float().to(device).unsqueeze(1)

        # predicted rewards (1 -> pred1 better than pred2, -1 -> the other way around)
        pred_r1s = reward_net(states1)
        pred_r2s = reward_net(states2)
        outputs = (pred_r1s - pred_r2s).view(-1)

        # pairwise ranking loss
        loss = -torch.mean(torch.log(torch.sigmoid(outputs * prefs.view(-1))))
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

save_file = '../results/' + args.env + '/' + args.reward + '/' + args.data_file + '.pt'
torch.save(reward_net, save_file)
print('Finished Training')
plt.plot(losses)
plt.savefig('../results/' + args.env + '/' + args.reward + '/' + args.data_file + '_losses.png')