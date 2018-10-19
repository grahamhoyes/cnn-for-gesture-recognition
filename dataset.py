'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

import torch.utils.data as data
import numpy as np
from utils import Util
util = Util()

def noisy(features, epsilon=1e-2):
    np.random.seed(util.get_cfg_param('seed'))
    noise = np.random.normal(scale=epsilon, size=features.shape)
    noisy_features = features + noise
    return noisy_features

class GestureDataset(data.Dataset):
    def __init__(self, X, y, add_noise=False, noise_epsilon=1e-2, noisy_samples=1e5):
        self.X = [X]
        self.y = [y]

        if add_noise:
            while noisy_samples > 0:
                noisy_features = noisy(X, epsilon=noise_epsilon)
                self.X.append(noisy_features)
                self.y.append(y)

                noisy_samples -= len(noisy_features)

        self.X = np.array(self.X).reshape((-1, *X.shape[1:]))
        self.y = np.array(self.y).reshape((-1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        instances = self.X[idx]
        label = self.y[idx]

        return instances, label