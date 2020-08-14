from pathlib import Path
import requests

import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np

import torch
import math

from pdb import set_trace

import torch.nn.functional as F


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open('wb').write(content)


with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding='latin-1'
    )


plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
plt.show()


x_train, y_train, x_valid, y_valid = map(
    torch.tensor,
    (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape


weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return xb @ weights + bias


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


bs = 64

xb = x_train[0: bs]
preds = model(xb)

loss_func = F.cross_entropy

yb = y_train[0: bs]
print(loss_func(preds, yb))
print(accuracy(preds, yb))
