'''
    Write a model for gesture classification.
'''

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv1d(128, 200, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=5, stride=1, padding=0)

        self.activation = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        # Fully connected layers
        self.fc1 = torch.nn.Linear(2600, 100)
        self.fc2 = torch.nn.Linear(100, 26)

    def forward(self, features):
        x = self.activation(self.conv1(features))
        # x = self.activation(x)
        x = self.activation(self.conv2(x))
        # x = self.activation(x)
        x = self.activation(self.conv3(x))
        # x = self.activation(x)
        # x = self.pool1(x)
        x = self.activation(self.conv4(x))
        # x = self.activation(x)
        # x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x