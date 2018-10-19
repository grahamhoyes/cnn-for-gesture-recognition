'''
    Write a model for gesture classification.
'''

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.activation = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        # Fully connected layers
        self.fc1 = torch.nn.Linear(1600, 128)
        self.fc2 = torch.nn.Linear(128, 26)

    def forward(self, features):
        x = self.activation(self.conv1(features))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(x)
        return x