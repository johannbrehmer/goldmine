from collections import OrderedDict
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, roc_curve

import torch
from torch import tensor, nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from goldmine.various.settings import dtype, dtype_np


def discriminate_samples(x1, x2, test_split=0.5,
                         n_hidden_layers=3, n_units_per_hidden_layer=100,
                         n_epochs=50, batch_size=64, initial_learning_rate=0.001, final_learning_rate=0.0001,
                         run_on_gpu=True):
    logging.info('Training classifier to discriminate two samples')

    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")

    # Prepare data
    xs = np.vstack([x1, x2]).astype(dtype_np)
    ys = np.hstack([np.zeros((x1.shape[0],)), np.ones((x2.shape[0],))]).astype(dtype_np)

    n_samples, n_inputs = xs.shape

    xs = torch.stack([tensor(i, dtype=dtype) for i in xs])
    ys = tensor(ys, dtype=torch.long)  # Has to be long to use CrossEntropyLoss

    dataset = TensorDataset(xs, ys)

    if test_split is not None:
        indices = list(range(n_samples))
        split = int(np.floor(test_split * n_samples))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_loader = DataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_idx),
            batch_size=batch_size,
            pin_memory=run_on_gpu
        )
        x_test = xs[valid_idx]
        y_test = ys[valid_idx]
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=run_on_gpu
        )
        x_test = xs
        y_test = ys

    # Make classifier
    layers = []

    for i in range(n_hidden_layers):
        n_in = n_inputs if i == 0 else n_units_per_hidden_layer
        n_out = 2 if i == n_hidden_layers - 1 else n_units_per_hidden_layer

        layers.append(('linear' + str(i), nn.Linear(n_in, n_out, 5)))

        if i < n_hidden_layers - 1:
            layers.append(('relu' + str(i), nn.ReLU()))

    layers.append(('softmax', nn.Softmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(layers))

    # Train
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate)

    for epoch in range(n_epochs):
        classifier.train()
        train_loss = 0.

        lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(epoch / (n_epochs - 1.))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i_batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            yhat = classifier(x)
            loss = loss_function(yhat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if (epoch + 1) % 10 == 0:
            logging.info('  Epoch %s / %s: train loss %s', epoch + 1, n_epochs, train_loss)

    # Evaluate
    logging.info('Evaluating discriminative classifier')
    classifier.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        yhat_test = classifier(x_test)

    y_test = y_test.detach().numpy()
    yhat_test = yhat_test.detach().numpy()[:, 1]

    # Calculate ROC AUC
    logging.info('Calculating ROC curve')
    fpr, tpr, _ = roc_curve(y_test, yhat_test)
    roc_auc = roc_auc_score(y_test, yhat_test)

    logging.info('ROC AUC: %s', roc_auc)

    return roc_auc, tpr, fpr
