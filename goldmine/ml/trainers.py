import logging
import numpy as np

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def train(model,
          loss,
          thetas, xs, ys=None,
          batch_size=64,
          initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
          run_on_gpu=True,
          validation_split=None, early_stopping=False, early_stopping_patience=10,
          learning_curve_folder=None, learning_curve_filename=None):
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

    # TODO: Implement early stopping
    # TODO: Save learning curves

    logging.info('Starting training')

    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")

    # Data set
    thetas = torch.stack([Tensor(i) for i in thetas])
    xs = torch.stack([Tensor(i) for i in xs])
    thetas_x = torch.stack([thetas, xs], dim=0)
    if ys is not None:
        ys = torch.stack([Tensor(i) for i in ys])

    dataset = TensorDataset(thetas_x, ys)

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

    # Log losses over training
    train_losses = []
    validation_losses = []

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
        for i_batch, (theta_x, y) in enumerate(train_loader):
            theta = theta_x[0, :]
            x = theta_x[1, :]
            theta = theta.to(device)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # Evaluate loss
            yhat = model(theta, x)
            loss = loss(model, y, yhat)
            train_loss += loss.item()

            # Calculate gradient and update optimizer
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss)

        # Validation
        if validation_split is None:
            if (epoch + 1) % 10 == 0:
                logging.info('  Epoch %d: train loss %.3f'
                             % (epoch + 1, train_losses[-1]))
            continue

        val_loss = 0.0

        with torch.no_grad():
            model.eval()

            for i_batch, (theta_x, y) in enumerate(validation_loader):
                theta = theta_x[0, :]
                x = theta_x[1, :]
                theta = theta.to(device)
                x = x.to(device)
                y = y.to(device)

                # Evaluate loss
                yhat = model(theta, x)
                loss = loss(model, y, yhat)
                val_loss += loss.item()

            validation_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            logging.info('  Epoch %d: train loss %.3f, validation loss %.3f'
                         % (epoch + 1, train_losses[-1], validation_losses[-1]))

        # ### OLD CODE ###
        #
        # theta_train, x_train = shuffle(theta_train, x_train)
        # theta_validation, x_validation = shuffle(theta_train, x_train)
        #
        # # Train batches
        # n_batches = int(math.ceil(len(x_train) / batch_size))
        # for i in range(n_batches):
        #     x_train_batch = Variable(torch.Tensor(x_train[i * batch_size:(i + 1) * batch_size]))
        #     theta_train_batch = Variable(torch.Tensor(theta_train[i * batch_size:(i + 1) * batch_size]))
        #     x_val_batch = Variable(torch.Tensor(x_validation[i * batch_size:(i + 1) * batch_size]))
        #     theta_val_batch = Variable(torch.Tensor(theta_validation[i * batch_size:(i + 1) * batch_size]))
        #
        #     optimizer.zero_grad()
        #
        #     u = model(theta_train_batch, x_train_batch)
        #     log_likelihood = model.log_likelihood
        #     loss = - torch.mean(log_likelihood)
        #     train_loss += loss.data[0]
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     u = model(theta_val_batch, x_val_batch)
        #     log_likelihood = model.log_likelihood
        #     loss = - torch.mean(log_likelihood)
        #     val_loss += loss.data[0]
        #
        # train_losses.append(train_loss)
        # validation_losses.append(val_loss)
        #
        # # print statistics
        # if epoch % 100 == 99:
        #     logging.info('Epoch %d: train loss %.3f, validation loss %.3f'
        #                  % (epoch + 1, train_losses[-1], validation_losses[-1]))
        #
        # #### GOOGLE CODE ###
        #
        # def train(epoch):
        #     model.train()
        #     train_loss = 0
        #     for batch_idx, (data, _) in enumerate(train_loader):
        #         data = data.to(device)
        #         optimizer.zero_grad()
        #         recon_batch, mu, logvar = model(data)
        #         loss = loss_function(recon_batch, data, mu, logvar)
        #         loss.backward()
        #         train_loss += loss.item()
        #         optimizer.step()
        #         if batch_idx % args.log_interval == 0:
        #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 epoch, batch_idx * len(data), len(train_loader.dataset),
        #                        100. * batch_idx / len(train_loader),
        #                        loss.item() / len(data)))
        #
        #     print('====> Epoch: {} Average loss: {:.4f}'.format(
        #         epoch, train_loss / len(train_loader.dataset)))
        #
        # def test(epoch):
        #     model.eval()
        #     test_loss = 0
        #     with torch.no_grad():
        #         for i, (data, _) in enumerate(test_loader):
        #             data = data.to(device)
        #             recon_batch, mu, logvar = model(data)
        #             test_loss += loss_function(recon_batch, data, mu, logvar).item()
        #             if i == 0:
        #                 n = min(data.size(0), 8)
        #                 comparison = torch.cat([data[:n],
        #                                         recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        #                 save_image(comparison.cpu(),
        #                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        #
        #     test_loss /= len(test_loader.dataset)
        #     print('====> Test set loss: {:.4f}'.format(test_loss))
        #
        # ### END ###

    logging.info('Finished training')

    return train_losses, validation_losses
