'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import string
import torch
import time

from utils import Util
from dataset import GestureDataset

from model import Net

# Load the dataset
util = Util()
_instances, _labels = util.load_norm_dataset()

# Perform the training / validation split
X_train, X_val, y_train, y_val = train_test_split(_instances, _labels, test_size=util.get_cfg_param('validation_split'),
                                                  random_state=util.get_cfg_param('seed'))

np.save('train_data.npy', X_train)
np.save('train_labels.npy', y_train)
np.save('val_data.npy', X_val)
np.save('val_labels.npy', y_val)

label_encoder = LabelEncoder()

def load_data(batch_size):
    train_set = GestureDataset(X_train, y_train, add_noise=True)
    val_set = GestureDataset(X_val, y_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader


def load_model(lr):
    model = Net()
    # loss_func = torch.nn.BCELoss()
    loss_func = torch.nn.MSELoss(reduction="elementwise_mean")
    # loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.NLLLoss()
    # loss_func = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    return model, loss_func, optimizer


def evaluate(model, val_loader, loss_func, deviceType="cpu"):
    device = torch.device(deviceType)

    total_corr = 0.0
    total_loss = 0.0

    for i, vbatch in enumerate(val_loader):
        features, labels = vbatch
        features = features.transpose(1, 2).to(device)
        labels = torch.Tensor(label_binarize(labels, classes=list(string.ascii_lowercase))).double().to(device)
        # labels = torch.Tensor(label_encoder.fit_transform(labels)).double().to(device)

        predictions = model(features)

        total_loss += loss_func(predictions.squeeze(), labels).item()

        total_corr += torch.sum(labels.argmax(dim=1) == predictions.argmax(dim=1)).item()
        # total_corr += torch.sum(labels == predictions.argmax(dim=1)).item()

    return float(total_corr)/len(val_loader.dataset), total_loss


def plot(model, train_accuracy, val_accuracy):
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    assert len(train_accuracy) == len(val_accuracy)
    steps = [i for i in range(len(train_accuracy))]
    ax.plot(steps, train_accuracy, label='Training accuracy')
    ax.plot(steps, val_accuracy, label='Validation accuracy')
    ax.grid()
    ax.legend()
    ax.set_title('Batch size %d | Learning rate %f | Epochs %d\nMax Validation Accuracy %f'
                 % (model.batch_size, model.learning_rate, model.epochs, max(val_accuracy)))
    plt.savefig(os.path.join(model.get_pathname(), 'accuracy_%s.png' % model.get_filename()))


def main():
    deviceType = "cuda" if torch.cuda.is_available() and util.get_cfg_param('cuda') else "cpu"
    device = torch.device(deviceType)
    print("Device type: %s" % deviceType)

    torch.manual_seed(util.get_cfg_param('seed'))

    train_loader, val_loader = load_data(util.get_cfg_param('batch_size'))
    model, loss_func, optimizer = load_model(util.get_cfg_param('learning_rate'))
    model = model.double().to(device) # Switch to cuda if used

    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    start_time = time.time()

    try:
        for epoch in range(util.get_cfg_param('epochs')):
            total_train_loss = 0
            total_corr = 0.0
            total_attempts = 0.0

            for i, batch in enumerate(train_loader):
                features, labels = batch
                features = features.transpose(1, 2).to(device)

                optimizer.zero_grad()
                outputs = model(features)

                labels = torch.Tensor(label_binarize(labels, classes=list(string.ascii_lowercase))).double().to(device)
                # labels = torch.Tensor(label_encoder.fit_transform(labels)).double().to(device)

                m = torch.nn.LogSoftmax()

                batch_loss = loss_func(outputs.squeeze(), labels)
                # batch_loss = loss_func(m(outputs), labels)
                total_train_loss += batch_loss
                batch_loss.backward()

                optimizer.step()

                total_corr += torch.sum(labels.argmax(dim=1) == outputs.argmax(dim=1)).item()
                # total_corr += torch.sum(labels == outputs.argmax(dim=1)).item()
                total_attempts += util.get_cfg_param('batch_size')
                total_train_loss += batch_loss.item()

            train_accuracy.append(float(total_corr)/float(total_attempts))
            train_loss.append(float(total_train_loss) / (i+1))

            batch_val_accuracy, batch_val_loss = evaluate(model, val_loader, loss_func, deviceType=deviceType)
            val_accuracy.append(batch_val_accuracy)
            val_loss.append(batch_val_loss)

            print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
                  % (epoch, train_accuracy[-1], train_loss[-1], val_accuracy[-1], val_loss[-1]))
    except KeyboardInterrupt:
        print('Stopping training')
        pass

    print('Finished training')
    end_time = time.time()
    print("total time elapsed: %0.2f" % (end_time - start_time))
    model.save()
    model.record(max(val_accuracy))

    plot(model, train_accuracy, val_accuracy)


if __name__ == "__main__":
    main()
