import logging
import torch
import torch.nn as nn

from goldmine.various.utils import get_activation_function


class LocalScoreEstimator(nn.Module):
    def __init__(self, n_observables, n_parameters, n_hidden, activation='tanh'):

        super(LocalScoreEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(
                nn.Linear(n_last, n_hidden_units)
            )
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(
            nn.Linear(n_last, n_parameters)
        )

    def forward(self, x):
        t_hat = x

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        return t_hat

    def to(self, *args, **kwargs):
        self = super(LocalScoreEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class CheckpointScoreEstimator(nn.Module):
    def __init__(self, n_latent, n_parameters, n_hidden, activation='tanh', delta_z_model=True):

        super(CheckpointScoreEstimator, self).__init__()

        # Save input
        self.n_latent = n_latent
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.delta_z_model = delta_z_model

        # Build network
        self.layers = nn.ModuleList()

        if self.delta_z_model:
            n_last = n_latent + n_parameters
        else:
            n_last = 2 * n_latent + n_parameters

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(
                nn.Linear(n_last, n_hidden_units)
            )
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(
            nn.Linear(n_last, n_parameters)
        )

    def forward(self, theta, z_initial, z_final):
        if self.delta_z_model:
            delta_z = z_final - z_initial
            t_hat = torch.cat((theta, delta_z), 1)
        else:
            t_hat = torch.cat((theta, z_initial, z_final), 1)

        logging.debug("Input to step network: %s", t_hat)

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        return t_hat

    def forward_trajectory(self, theta, z_checkpoints):
        # Prepare data
        n_batch, n_steps, n_latent = z_checkpoints.size()
        n_parameters = theta.size()[-1]

        logging.debug("theta: %s", theta)
        logging.debug("z_checkpoints: %s", z_checkpoints)

        z_initial = z_checkpoints[:, :-1, :].contiguous().view(-1, n_latent)
        z_final = z_checkpoints[:, 1:, :].contiguous().view(-1, n_latent)

        logging.debug("z_initial: %s", z_initial)
        logging.debug("z_final: %s", z_final)

        theta_step = theta.clone()
        theta_step.unsqueeze(1)  # (n_batch, 1, n_parameters)
        theta_step = theta_step.repeat(1, n_steps - 1, 1)
        theta_step = theta_step.contiguous().view(-1, n_parameters)

        logging.debug("theta_step: %s", theta_step)

        # Step model (score between checkpoints) for that_i(v_i, v_{i-1})
        that_xv_checkpoints = self.forward(theta_step, z_initial, z_final)
        logging.debug("that_xv_checkpoints (raw): %s", that_xv_checkpoints)
        that_xv_checkpoints = that_xv_checkpoints.view(n_batch, n_steps - 1, n_parameters).contiguous()
        logging.debug("that_xv_checkpoints (processes): %s", that_xv_checkpoints)

        return that_xv_checkpoints

    def to(self, *args, **kwargs):
        self = super(CheckpointScoreEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
