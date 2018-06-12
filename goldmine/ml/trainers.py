import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class GoldDataset(torch.utils.data.Dataset):

    def __init__(self, theta, x, y=None, r_xz=None, t_xz=None):
        self.n = theta.shape[0]

        placeholder = torch.stack([tensor([0.]) for i in range(self.n)])

        self.theta = theta
        self.x = x
        self.y = placeholder if y is None else y
        self.r_xz = placeholder if r_xz is None else r_xz
        self.t_xz = placeholder if t_xz is None else t_xz

        assert len(self.theta) == self.n
        assert len(self.x) == self.n
        assert len(self.y) == self.n
        assert len(self.r_xz) == self.n
        assert len(self.t_xz) == self.n

    def __getitem__(self, index):
        return (self.theta[index],
                self.x[index],
                self.y[index],
                self.r_xz[index],
                self.t_xz[index])

    def __len__(self):
        return self.n


def train(model,
          loss_function,
          thetas, xs, ys=None, r_xzs=None, t_xzs=None,
          batch_size=64,
          initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
          run_on_gpu=True,
          validation_split=0.2, early_stopping=True,
          learning_curve_folder=None, learning_curve_filename=None,
          n_epochs_verbose=1):
    """

    :param model:
    :param loss: function loss(model, y_true, y_pred)
    :param thetas:
    :param xs:
    :param ys:
    :param batch_size:
    :param initial_learning_rate:
    :param final_learning_rate:
    :param n_epochs:
    :param run_on_gpu:
    :param validation_split:
    :param early_stopping:
    :param early_stopping_patience:
    :param learning_curve_folder:
    :param learning_curve_filename:
    :return:
    """

    # TODO: support for r and z terms in losses

    logging.info('Starting training')

    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")

    # Convert to Tensor
    thetas = torch.stack([tensor(i, requires_grad=True) for i in thetas])
    xs = torch.stack([tensor(i) for i in xs])
    if ys is not None:
        ys = torch.stack([tensor(i) for i in ys])
    if r_xzs is not None:
        r_xzs = torch.stack([tensor(i) for i in r_xzs])
    if t_xzs is not None:
        t_xzs = torch.stack([tensor(i) for i in t_xzs])

    # Dataset
    dataset = GoldDataset(thetas, xs, ys, r_xzs, t_xzs)

    # Train / validation split
    if validation_split is not None:
        assert 0. < validation_split < 1., 'Wrong validation split: {}'.format(validation_split)

        n_samples = len(dataset)
        indices = list(range(n_samples))
        split = int(np.floor(validation_split * n_samples))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            pin_memory=run_on_gpu
        )
        validation_loader = DataLoader(
            dataset,
            sampler=validation_sampler,
            batch_size=batch_size,
            pin_memory=run_on_gpu
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=run_on_gpu
        )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # Early stopping
    early_stopping = early_stopping and (validation_split is not None)
    early_stopping_best_val_loss = None
    early_stopping_best_model = None
    early_stopping_epoch = None

    # Log losses over training
    train_losses = []
    val_losses = []

    # Loop over epochs
    for epoch in range(n_epochs):

        # Training
        model.train()
        train_loss = 0.0

        # Learning rate decay
        lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(epoch / (n_epochs - 1.))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Loop over batches
        for i_batch, (theta, x, y, r_xz, t_xz) in enumerate(train_loader):
            theta = theta.to(device)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Evaluate loss
            yhat = model(theta, x)
            loss = loss_function(model, y, yhat)
            train_loss += loss.item()

            # Calculate gradient and update optimizer
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss)

        # Validation
        if validation_split is None:
            if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
                logging.info('  Epoch %d: train loss %.3f'
                             % (epoch + 1, train_losses[-1]))
            continue

        val_loss = 0.0

        # with torch.no_grad():
        model.eval()

        for i_batch, (theta, x, y, r_xz, t_xz) in enumerate(validation_loader):
            theta = theta.to(device)
            x = x.to(device)
            y = y.to(device)

            # Evaluate loss
            yhat = model(theta, x)
            loss = loss_function(model, y, yhat)
            val_loss += loss.item()

        val_losses.append(val_loss)

        # Early stopping
        if early_stopping:
            if early_stopping_best_val_loss is None or val_loss < early_stopping_best_val_loss:
                early_stopping_best_val_loss = val_loss
                early_stopping_best_model = model.state_dict()
                early_stopping_epoch = epoch

        if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
            if early_stopping and epoch == early_stopping_epoch:
                logging.info('  Epoch %d: train loss %.3f, validation loss %.3f (*)'
                             % (epoch + 1, train_losses[-1], val_losses[-1]))
            else:
                logging.info('  Epoch %d: train loss %.3f, validation loss %.3f'
                             % (epoch + 1, train_losses[-1], val_losses[-1]))

    # Early stopping
    if early_stopping:
        if early_stopping_best_val_loss < val_loss:
            logging.info('Early stopping after epoch %s, with loss %s compared to final loss %s',
                         early_stopping_epoch + 1, early_stopping_best_val_loss, val_loss)
            model.load_state_dict(early_stopping_best_model)
        else:
            logging.info('Early stopping did not improve performance')

    # Save learning curve
    if learning_curve_folder is not None and learning_curve_filename is not None:
        np.save(learning_curve_folder + '/' + learning_curve_filename + '_train_loss.npy', train_losses)
        if validation_split is not None:
            np.save(learning_curve_folder + '/' + learning_curve_filename + '_validation_loss.npy', val_losses)

    logging.info('Finished training')

    return train_losses, val_losses
