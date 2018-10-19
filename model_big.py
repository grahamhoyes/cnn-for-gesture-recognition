'''
    Write a model for gesture classification.
'''

import torch
import torch.nn as nn
from utils import Util
import os
import json
from shutil import copy2

util = Util()

class Net(nn.Module):
    def __init__(self):
        # Create the folder that it will be saved to
        path = util.get_folder()
        self.pathname = util.make_folder(path)
        self.filename = util.get_folder()

        # Initialize some parameters
        self.batch_size = util.get_cfg_param('batch_size')
        self.learning_rate = util.get_cfg_param('learning_rate')
        self.epochs = util.get_cfg_param('epochs')

        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=0)
        self.conv6 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=0)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=9, stride=1, padding=0)
        self.conv8 = nn.Conv1d(256, 64, kernel_size=9, stride=3, padding=0)
        self.conv9 = nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=0)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.leaky = nn.LeakyReLU(0.1)
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # Drop out of engsci
        self.dropout = nn.Dropout(0.1)

        # Fully connected layers
        self.fc1 = torch.nn.Linear(4*32, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 26)

    def forward(self, features):
        x = self.elu(self.conv1(features))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.elu(self.conv6(x))
        x = self.elu(self.conv7(x))
        x = self.elu(self.conv8(x))
        x = self.elu(self.conv9(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.prelu(self.fc1(x))
        x = self.prelu(self.fc2(x))
        x = self.prelu(self.fc3(x))

        return x

    def save(self):
        # Save the pytorch model
        torch.save(self, os.path.join(self.pathname, 'model.pt'))
        print('Model saved to %s' % os.path.join(self.pathname, 'model.pt'))

        # Save model.py
        copy2('model.py', os.path.join(self.pathname, 'model.py'))
        print('model.py saved to %s' % os.path.join(self.pathname, 'model.py'))

    def record(self, max_val_accuracy):
        new_data = {'best:': self.pathname, 'max_val_accuracy': max_val_accuracy}

        if not os.path.isfile('models.json'):
            with open('models.json', 'w+') as fh:
                json.dump(new_data, fh)
        else:
            with open('models.json', 'r') as fh:
                data = json.load(fh)
            if data['max_val_accuracy'] < max_val_accuracy:
                with open('models.json', 'w') as fh:
                    json.dump(new_data, fh)

    def get_pathname(self):
        return self.pathname

    def get_filename(self):
        return self.filename