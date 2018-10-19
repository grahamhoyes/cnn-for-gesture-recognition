'''
    Normalize the data, save as ./data/normalized_data.npy
'''

import numpy as np

instances = np.load('instances.npy')
instances_normalized = []


for i in range(len(instances)):
    mean = instances[i].mean(axis=0)
    stdev = instances[i].std(axis=0)

    norm = (instances[i] - mean)/stdev
    instances_normalized.append(norm)

np.save('data/normalized_data.npy', np.array(instances_normalized))