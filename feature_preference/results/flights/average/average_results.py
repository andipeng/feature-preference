import glob
import ast
import numpy as np

files = glob.glob("../**/results.txt", recursive=True)

results = {}
for file in files:
    with open(file, "r") as f:
        content = f.read().splitlines()

    for line in content:
        key, value = line.split(' = ')
        value = ast.literal_eval(value)

        if key not in results:
            results[key] = []

        results[key].append(value)

averaged_results = {}
for key, values in results.items():
    if key == 'comparisons':
        averaged_results['comparisons'] = ['train_1', 'train_3', 'train_5', 'train_10']
    else:
        averaged_results[key] = np.mean(values, axis=0).tolist()

with open('results_averaged.txt', 'w') as f:
    for key, value in averaged_results.items():
        f.write(f'{key} = {value}\n')