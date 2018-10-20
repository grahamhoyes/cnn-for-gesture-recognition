from utils import Util
import numpy as np
import json
import torch
import os

def grade():
    with open('models.json', 'r') as fh:
        models = json.load(fh)

    modelpath = os.path.join(models['best:'], 'model.pt')
    model = torch.load(modelpath)

    dataset = np.load('test_data/test_data.npy').swapaxes(1, 2)
    data = torch.Tensor(dataset).double().cuda()

    predictions = model(data)
    return predictions

def test(model, instances):
    pass

if __name__ == "__main__":
    a = grade()