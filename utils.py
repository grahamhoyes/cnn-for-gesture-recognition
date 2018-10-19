import json
import numpy as np
import os

class Util:
    def __init__(self):
        # Load the config json file
        with open('config.json', 'r') as fh:
            self.config = json.load(fh)

    def get_cfg_param(self, param):
        if param in self.config:
            return self.config[param]
        else:
            return None

    def set_cfg_param(self, param, value):
        self.config[param] = value
        with open('config.json', 'w') as fh:
            json.dump(self.config, fh)

    def load_dataset(self):
        instances = np.load('instances.npy')
        labels = np.load('labels.npy')

        return instances, labels

    def load_norm_dataset(self):
        instances = np.load('data/normalized_data.npy')
        labels = np.load('labels.npy')

        return instances, labels

    def get_folder(self):
        filename = 'bs%d_lr%s_epochs%d_seed%d%s' \
                   % (self.config['batch_size'], str(self.config['learning_rate']).replace('.', '-'),
                      self.config['epochs'], self.config['seed'], self.config['notes'])
        return filename

    def make_folder(self, filename):
        i = 0
        while os.path.isdir(os.path.join('models', 'model_%s_%d' % (filename, i))):
            i += 1
        pathname = os.path.join('models', 'model_%s_%d' % (filename, i))
        os.mkdir(pathname)

        return pathname