'''
    The entry into your code. This file should include a training function and an evaluation function.
'''
import argparse
import numpy as np
import matplotlib as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from time import time

import torch
from model import Net
from torch.utils.data import DataLoader
from dataset import GestureDataset

def load_data(batch_size):
    norm_instances = np.load("data/normalized_data.npy")
    labels = np.load("data/labels.npy")

    # split data
    (train_X, valid_X, train_y, valid_y) = train_test_split(norm_instances, labels, test_size=0.2, random_state=0)

    #load data
    train_dataset = GestureDataset(train_X, train_y)
    valid_dataset = GestureDataset(valid_X, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, val_loader


def load_model(lr, input_size=6):
    model = Net(input_size)
    loss_fnc = torch.nn.functional.binary_cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return model, loss_fnc, optimizer


def evaluate(valid_loader, model, criterion):
    valid_loss_total = 0.0
    valid_err_total = 0

    for data in valid_loader:
        (valid_X, valid_Y) = data

        outputs = model(valid_X.float())
        loss = criterion(valid_Y, outputs)
        valid_loss_total += loss.item()

        valid_err_total += int(torch.sum(outputs.argmax(dim=1) != valid_Y.argmax(dim=1)))

    return (valid_loss_total / len(valid_loader.dataset), float(valid_err_total) / len(valid_loader.dataset))


def train(bs=64, lr=0.01, epochs=100):
    (train_loader, valid_loader) = load_data(bs)
    (model, loss_fnc, optimizer) = load_model(lr)

    train_acc = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)

    start_time = time()

    for epoch in range(epochs):
        total_train_acc = 0.0
        total_train_loss = 0.0
        total_epoch = 0.0

        for (i, data) in enumerate(train_loader, 0):
            (train_X, train_y) = data

            optimizer.zero_grad()
            outputs = model(train_X.float())
            loss = loss_fnc(train_y, outputs)
            loss.backward()
            optimizer.step()

            # Calculate the statistics
            total_train_acc += int(torch.sum(outputs.argmax(dim=1) != train_Y.argmax(dim=1)))
            total_train_loss += loss.item()
            total_epoch += bs

        train_acc[epoch] = 1 - float(total_train_acc)/total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_acc[epoch] = evaluate(model, valid_loader, loss_fnc)

        print("Epoch {}: Train acc: {}, Train loss: {} | Validation accuracy: {}".format(epoch + 1, train_acc[epoch], train_loss[epoch], val_acc[epoch]))
    #print(len(train_acc_fine))
    print(time() - start_time, "seconds")
    print("Finished training.")
    return train_acc, train_loss, val_acc
    ######

if __name__ == "__main__":
    train()
