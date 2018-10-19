'''
    Write a model for gesture classification.
'''

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(6, 100, kernel_size=10, stride=1, padding=0)
        self.conv2 = nn.Conv1d(100, 200, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv1d(200, 50, kernel_size=3, stride=1, padding=0)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=5, padding=0)

        self.conv4 = nn.Conv1d(50, 100, kernel_size=6, stride=1, padding=0)
        self.conv5 = nn.Conv1d(100, 150, kernel_size=11, stride=1, padding=0)

        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=1, padding=0)

        self.activation = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        # Fully connected layers
        self.fc1 = torch.nn.Linear(150, 50)
        self.fc2 = torch.nn.Linear(50, 26)

    def forward(self, features):
        x = self.activation(self.conv1(features))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.maxpool(x)
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(x)
        return x