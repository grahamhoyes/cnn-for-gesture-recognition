'''
    Normalize the data, save as ./data/normalized_data.npy
'''
import numpy as np

def normalize(instances, output):

    instances = np.load(instances)
    instances_normalized = []


    for i in range(len(instances)):
        mean = instances[i].mean(axis=0)
        stdev = instances[i].std(axis=0)

        norm = (instances[i] - mean)/stdev
        instances_normalized.append(norm)

    np.save(output, np.array(instances_normalized))

if __name__ == "__main__":
    normalize('instances.npy', 'data/normalized_data.npy')