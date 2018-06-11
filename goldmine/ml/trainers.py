import math
import logging
import numpy as np
import numpy.random as rng
from scipy.stats import beta, norm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def simple_trainer(model,
                   loss,
                   inputs, labels=None,
                   batch_size=64,
                   initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
                   validation_split=None, early_stopping=False, early_stopping_patience=10,
                   learning_curve_folder=None, learning_curve_filename=None):

    # Data loader
    xs = torch.stack([torch.Tensor(i) for i in inputs])
    if y is None:
        ys = None
    else:
        ys = torch.stack([torch.Tensor(i) for i in labels])

    dataset = TensorDataset(xs, ys)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # LR decay:
    def adjust_lr(epoch):
        lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(epoch / (n_epochs-1.))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    # somewhere here I am



    train_losses = []
    validation_losses = []

    for epoch in range(n_epochs):

        train_loss = 0.0
        val_loss = 0.0

        theta_train, x_train = shuffle(theta_train, x_train)
        theta_validation, x_validation = shuffle(theta_train, x_train)

        # Train batches
        n_batches = int(math.ceil(len(x_train) / batch_size))
        for i in range(n_batches):
            x_train_batch = Variable(torch.Tensor(x_train[i * batch_size:(i + 1) * batch_size]))
            theta_train_batch = Variable(torch.Tensor(theta_train[i * batch_size:(i + 1) * batch_size]))
            x_val_batch = Variable(torch.Tensor(x_validation[i * batch_size:(i + 1) * batch_size]))
            theta_val_batch = Variable(torch.Tensor(theta_validation[i * batch_size:(i + 1) * batch_size]))

            optimizer.zero_grad()

            u = model(theta_train_batch, x_train_batch)
            log_likelihood = model.log_likelihood
            loss = - torch.mean(log_likelihood)
            train_loss += loss.data[0]

            loss.backward()
            optimizer.step()

            u = model(theta_val_batch, x_val_batch)
            log_likelihood = model.log_likelihood
            loss = - torch.mean(log_likelihood)
            val_loss += loss.data[0]

        train_losses.append(train_loss)
        validation_losses.append(val_loss)

        # print statistics
        if epoch % 100 == 99:
            logging.info('Epoch %d: train loss %.3f, validation loss %.3f'
                  % (epoch + 1, train_losses[-1], validation_losses[-1]))

    logging.info('Finished Training')

    return train_losses, validation_losses