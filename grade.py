from utils import Util
import numpy as np
import json
import torch
import os
from shutil import copy2
from normalize_data import normalize

def normalize_test_data():
    normalize('test_data/test_data.npy', 'test_data/test_data_normalized.npy')

def grade():
    with open('models.json', 'r') as fh:
        models = json.load(fh)

    # Copy model.py
    copy2('model.py', 'model.py.bak')
    copy2(os.path.join(models['best:'], 'model.py'), 'model.py')

    modelpath = os.path.join(models['best:'], 'model.pt')
    model = torch.load(modelpath)

    dataset = np.load('test_data/test_data_normalized.npy').swapaxes(1, 2)
    data = torch.Tensor(dataset).double().cuda()

    predictions = model(data)
    predictions = predictions.argmax(dim=1)
    np.savetxt('predictions.txt', predictions)

    # Restore model.py
    os.remove('model.py')
    copy2('model.py.bak', 'model.py')
    os.remove('model.py.bak')

    return predictions


if __name__ == "__main__":
    normalize_test_data()
    grade()