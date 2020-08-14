from pathlib import Path
import requests

import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np

import torch
import math

from pdb import set_trace


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
    return log_softmax(xb @ weights + bias)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


bs = 64

xb = x_train[0: bs]
preds = model(xb)

loss_func = nll

yb = y_train[0: bs]
print(loss_func(preds, yb))
print(accuracy(preds, yb))


lr = 0.5
epochs = 20

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb).item(), accuracy(model(xb), yb).item())
