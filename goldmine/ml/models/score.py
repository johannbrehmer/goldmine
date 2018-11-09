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
    def __init__(self, n_latent, n_parameters, n_hidden, activation='tanh'):

        super(CheckpointScoreEstimator, self).__init__()

        # Save input
        self.n_latent = n_latent
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
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
        t_hat = torch.cat((theta, z_initial, z_final), 1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        return t_hat

    def to(self, *args, **kwargs):
        self = super(CheckpointScoreEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
