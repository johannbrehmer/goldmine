import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from goldmine.various.utils import check_for_nans_in_parameters


class CheckpointedGoldDataset(torch.utils.data.Dataset):

    def __init__(self, theta, x, y=None, r_xz=None, t_xz=None, z_checkpoints=None, r_xz_checkpoints=None,
                 t_xz_checkpoints=None):
        self.n = theta.shape[0]

        placeholder = torch.stack([tensor([0.]) for _ in range(self.n)])

        self.theta = theta
        self.x = x
        self.y = placeholder if y is None else y
        self.r_xz = placeholder if r_xz is None else r_xz
        self.t_xz = placeholder if t_xz is None else t_xz
        self.z_checkpoints = placeholder if z_checkpoints is None else z_checkpoints
        self.r_xz_checkpoints = placeholder if r_xz_checkpoints is None else r_xz_checkpoints
        self.t_xz_checkpoints = placeholder if t_xz_checkpoints is None else t_xz_checkpoints

        assert len(self.theta) == self.n
        assert len(self.x) == self.n
        assert len(self.y) == self.n
        assert len(self.r_xz) == self.n
        assert len(self.t_xz) == self.n
        assert len(self.z_checkpoints) == self.n
        assert len(self.r_xz_checkpoints) == self.n
        assert len(self.t_xz_checkpoints) == self.n

    def __getitem__(self, index):
        return (self.theta[index],
                self.x[index],
                self.y[index],
                self.r_xz[index],
                self.t_xz[index],
                self.z_checkpoints[index],
                self.r_xz_checkpoints[index],
                self.t_xz_checkpoints[index])

    def __len__(self):
        return self.n


def train_checkpointed_model(
        model,
        score_model,
        loss_functions,
        thetas, xs, ys=None, r_xzs=None, t_xzs=None,
        theta1=None,
        z_checkpoints=None, r_xz_checkpoints=None, t_xz_checkpoints=None,
        step_mode='score',
        global_mode='flow',
        loss_weights=None,
        loss_labels=None,
        batch_size=64,
        trainer='adam',
        initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
        clip_gradient=10.,
        freeze_model=False,
        freeze_score_model=False,
        run_on_gpu=True,
        double_precision=False,
        validation_split=0.2, early_stopping=True, early_stopping_patience=None,
        learning_curve_folder=None, learning_curve_filename=None,
        verbose='some'

):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Move model to device
    model = model.to(device, dtype)
    score_model = score_model.to(device, dtype)

    # Convert to Tensor
    thetas = torch.stack([tensor(i, requires_grad=True) for i in thetas])
    xs = torch.stack([tensor(i) for i in xs])
    if ys is not None:
        ys = torch.stack([tensor(i.astype(np.float)) for i in ys])  # pyTorch cannot cast np.int64 or np.float32?
    if r_xzs is not None:
        r_xzs = torch.stack([tensor(i) for i in r_xzs])
    if t_xzs is not None:
        t_xzs = torch.stack([tensor(i) for i in t_xzs])
    if z_checkpoints is not None:
        z_checkpoints = torch.stack([tensor(i) for i in z_checkpoints])
    if r_xz_checkpoints is not None:
        r_xz_checkpoints = torch.stack([tensor(i) for i in r_xz_checkpoints])
    if t_xz_checkpoints is not None:
        t_xz_checkpoints = torch.stack([tensor(i) for i in t_xz_checkpoints])

    # Dataset
    dataset = CheckpointedGoldDataset(thetas, xs, ys, r_xzs, t_xzs, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints)

    # Mode
    assert global_mode in ['flow', 'ratio']
    assert step_mode in ['score', 'ratio']

    # Val split
    if validation_split is not None and validation_split <= 0.:
        validation_split = None

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

    # Hyperparameters to be optimized
    if freeze_model and freeze_score_model:
        raise ValueError("Cannot freeze both model and score model!")
    elif freeze_model:
        parameters = score_model.parameters()
    elif freeze_score_model:
        parameters = model.parameters()
    else:
        parameters = list(model.parameters()) + list(score_model.parameters())

    # Optimizer
    if trainer == 'adam':
        optimizer = optim.Adam(parameters, lr=initial_learning_rate)
    elif trainer == 'sgd':
        optimizer = optim.SGD(parameters, lr=initial_learning_rate)
    else:
        raise ValueError('Unknown trainer {}'.format(trainer))

    # Early stopping
    early_stopping = early_stopping and (validation_split is not None) and (n_epochs > 1)
    early_stopping_best_val_loss = None
    early_stopping_best_model = None
    early_stopping_epoch = None

    # Loss functions
    n_losses = len(loss_functions)

    if loss_weights is None:
        loss_weights = [1.] * n_losses

    # Losses over training
    individual_losses_train = []
    individual_losses_val = []
    total_losses_train = []
    total_losses_val = []
    total_val_loss = None

    log_r = None

    # Verbosity
    n_epochs_verbose = None
    if verbose == 'all':  # Print output after every epoch
        n_epochs_verbose = 1
    elif verbose == 'some':  # Print output after 10%, 20%, ..., 100% progress
        n_epochs_verbose = max(int(round(n_epochs / 10, 0)), 1)

    logging.info('Starting training')

    # Loop over epochs
    for epoch in range(n_epochs):

        # Training
        model.train()
        individual_train_loss = np.zeros(n_losses)
        total_train_loss = 0.0

        # Learning rate decay
        if n_epochs > 1:
            lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(epoch / (n_epochs - 1.))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Loop over batches
        for i_batch, batch_data in enumerate(train_loader):
            theta, x, y, r_xz, t_xz, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints = batch_data

            # Put on device
            theta = theta.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype).view(-1)
            try:
                r_xz = r_xz.to(device, dtype).view(-1)
            except NameError:
                pass
            try:
                t_xz = t_xz.to(device, dtype)
            except NameError:
                pass
            try:
                z_checkpoints = z_checkpoints.to(device, dtype)
            except NameError:
                pass
            try:
                r_xz_checkpoints = r_xz_checkpoints.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz_checkpoints = t_xz_checkpoints.to(device, dtype)
            except NameError:
                pass
            if theta1 is not None:
                theta1_tensor = torch.tensor(theta1).to(device, dtype)
                theta1_tensor = theta1_tensor.view(1, -1).expand_as(theta)

            optimizer.zero_grad()

            # Step model (score between checkpoints) for that_i(v_i, v_{i-1})
            that_xv_checkpoints = score_model.forward_trajectory(theta, z_checkpoints)

            # Sum for that(x,v)
            that_xv = torch.sum(that_xv_checkpoints, dim=1)

            # Global model (flow)
            if global_mode == 'flow':
                _, log_likelihood, that_x = model.log_likelihood_and_score(theta, x)

                if theta1 is not None:
                    _, log_likelihood_theta1 = model.model.log_likelihood(theta1_tensor, x)
                    log_r = log_likelihood - log_likelihood_theta1
            elif global_mode == 'ratio':
                raise NotImplementedError
                # _, log_r, score = model(theta, x, track_score=(t_xz is not None))
                # log_r = log_r.view(-1)
                # log_likelihood = None
            else:
                raise ValueError('Unknown method type {}'.format(global_mode))

            # Evaluate loss
            try:
                # Signature: fn(log_p_pred, t_x_pred, t_xv_pred, t_xv_checkpoints_pred, t_xz_checkpoints)
                losses = [fn(log_likelihood, that_x, that_xv, that_xv_checkpoints, t_xz_checkpoints) for fn in
                          loss_functions]
            except RuntimeError:
                logging.error('Error in evaluating loss functions!')
                raise

            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_train_loss[i] += individual_loss.item()
            total_train_loss += loss.item()

            # Calculate gradient
            loss.backward()

            # Clip gradients
            if clip_gradient is not None:
                clip_grad_norm_(model.parameters(), clip_gradient)

            # Check for NaNs
            if check_for_nans_in_parameters(model):
                logging.warning('NaNs in parameters or gradients, stopping training!')
                break

            # Optimizer step
            optimizer.step()

        individual_train_loss /= len(train_loader)
        total_train_loss /= len(train_loader)

        total_losses_train.append(total_train_loss)
        individual_losses_train.append(individual_train_loss)

        # Validation
        if validation_split is None:
            if (n_epochs_verbose is not None and n_epochs_verbose > 0 and
                    (epoch == 0 or (epoch + 1) % n_epochs_verbose == 0)):
                logging.info('  Epoch %d: train loss %.2f (%s)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1]))
            continue

        # with torch.no_grad():
        model.eval()
        individual_val_loss = np.zeros(n_losses)
        total_val_loss = 0.0

        for i_batch, batch_data in enumerate(validation_loader):
            theta, x, y, r_xz, t_xz, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints = batch_data

            # Put on device
            theta = theta.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype).view(-1)
            try:
                r_xz = r_xz.to(device, dtype).view(-1)
            except NameError:
                pass
            try:
                t_xz = t_xz.to(device, dtype)
            except NameError:
                pass
            try:
                z_checkpoints = z_checkpoints.to(device, dtype)
            except NameError:
                pass
            try:
                r_xz_checkpoints = r_xz_checkpoints.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz_checkpoints = t_xz_checkpoints.to(device, dtype)
            except NameError:
                pass
            if theta1 is not None:
                theta1_tensor = torch.tensor(theta1).to(device, dtype)
                theta1_tensor = theta1_tensor.view(1, -1).expand_as(theta)

            # Step model (score between checkpoints) for that_i(v_i, v_{i-1})
            that_xv_checkpoints = score_model.forward_trajectory(theta, z_checkpoints)

            # Sum for that(x,v)
            that_xv = torch.sum(that_xv_checkpoints, dim=1)

            # Global model (flow)
            if global_mode == 'flow':
                _, log_likelihood, that_x = model.log_likelihood_and_score(theta, x)

                if theta1 is not None:
                    _, log_likelihood_theta1 = model.model.log_likelihood(theta1_tensor, x)
                    log_r = log_likelihood - log_likelihood_theta1
            elif global_mode == 'ratio':
                raise NotImplementedError
                # _, log_r, score = model(theta, x, track_score=(t_xz is not None))
                # log_r = log_r.view(-1)
                # log_likelihood = None
            else:
                raise ValueError('Unknown method type {}'.format(global_mode))

            # Evaluate loss
            try:
                # Signature: fn(log_p_pred, t_x_pred, t_xv_pred, t_xv_checkpoints_pred, t_xz_checkpoints)
                losses = [fn(log_likelihood, that_x, that_xv, that_xv_checkpoints, t_xz_checkpoints) for fn in
                          loss_functions]
            except RuntimeError:
                logging.error('Error in evaluating loss functions!')
                raise

            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_val_loss[i] += individual_loss.item()
            total_val_loss += loss.item()

        individual_val_loss /= len(validation_loader)
        total_val_loss /= len(validation_loader)

        total_losses_val.append(total_val_loss)
        individual_losses_val.append(individual_val_loss)

        # Early stopping: best epoch so far?
        if early_stopping:
            if early_stopping_best_val_loss is None or total_val_loss < early_stopping_best_val_loss:
                early_stopping_best_val_loss = total_val_loss
                early_stopping_best_model = model.state_dict()
                early_stopping_epoch = epoch

        # Print out information
        if (n_epochs_verbose is not None and n_epochs_verbose > 0 and
                (epoch == 0 or (epoch + 1) % n_epochs_verbose == 0)):
            if early_stopping and epoch == early_stopping_epoch:
                logging.info('  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s) (*)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1],
                                total_losses_val[-1], individual_losses_val[-1]))
            else:
                logging.info('  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1],
                                total_losses_val[-1], individual_losses_val[-1]))

        # Early stopping: actually stop training
        if early_stopping and early_stopping_patience is not None:
            if epoch - early_stopping_epoch >= early_stopping_patience > 0:
                logging.info('No improvement for %s epochs, stopping training', epoch - early_stopping_epoch)
                break

    # Early stopping: back to best state
    if early_stopping:
        if early_stopping_best_val_loss < total_val_loss:
            logging.info('Early stopping after epoch %s, with loss %.2f compared to final loss %.2f',
                         early_stopping_epoch + 1, early_stopping_best_val_loss, total_val_loss)
            model.load_state_dict(early_stopping_best_model)
        else:
            logging.info('Early stopping did not improve performance')

    # Save learning curve
    if learning_curve_folder is not None and learning_curve_filename is not None:

        np.save(learning_curve_folder + '/loss_train_' + learning_curve_filename + '.npy', total_losses_train)
        if validation_split is not None:
            np.save(learning_curve_folder + '/loss_val_' + learning_curve_filename + '.npy', total_losses_val)

        if loss_labels is not None:
            individual_losses_train = np.array(individual_losses_train)
            individual_losses_val = np.array(individual_losses_val)

            for i, label in enumerate(loss_labels):
                np.save(
                    learning_curve_folder + '/loss_' + label + '_train' + learning_curve_filename + '.npy',
                    individual_losses_train[:, i]
                )
                if validation_split is not None:
                    np.save(
                        learning_curve_folder + '/loss_' + label + '_val' + learning_curve_filename + '.npy',
                        individual_losses_val[:, i]
                    )

    logging.info('Finished training')

    return total_losses_train, total_losses_val
