import numpy as np
import logging
import torch
from torch import tensor

from goldmine.inference.base import Inference
from goldmine.ml.models.ratio import ParameterizedRatioEstimator
from goldmine.various.utils import expand_array_2d


class RatioInference(Inference):

    def __init__(self, **params):
        super().__init__()

        filename = params.get('filename', None)

        if filename is None:
            # Parameters for new MAF
            n_parameters = params['n_parameters']
            n_observables = params['n_observables']
            n_hidden_layers = params.get('n_hidden_layers', 5)
            n_units_per_layer = params.get('n_units_per_layer', 100)
            activation = params.get('activation', 'tanh')

            logging.info('Initialized ratio regressor with the following settings:')
            logging.info('  Parameters:    %s', n_parameters)
            logging.info('  Observables:   %s', n_observables)
            logging.info('  Hidden layers: %s', n_hidden_layers)
            logging.info('  Units:         %s', n_units_per_layer)
            logging.info('  Activation:    %s', activation)

            # MAF
            self.regressor = ParameterizedRatioEstimator(
                n_parameters=n_parameters,
                n_observables=n_observables,
                n_hidden=tuple([n_units_per_layer] * n_hidden_layers),
                activation=activation
            )

        else:
            self.regressor = torch.load(filename + '.pt', map_location='cpu')

            logging.info('Loaded ratio regressor from file:')
            logging.info('  Hidden layers: %s', self.regressor.n_hidden)
            logging.info('  Activation:    %s', self.regressor.activation)

        # Have everything on CPU (unless training)
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def save(self, filename):
        # Fix a bug in pyTorch, see https://github.com/pytorch/text/issues/350
        self.regressor.to()

        torch.save(self.regressor, filename + '.pt')

        self.regressor.to(self.device, self.dtype)

    def predict_ratio(self, theta0, theta1, x, log=False):
        # If just one theta given, broadcast to number of samples
        theta0 = expand_array_2d(theta0, x.shape[0])
        theta1 = expand_array_2d(theta1, x.shape[0])

        self.regressor = self.regressor.to(self.device, self.dtype)
        theta0_tensor = tensor(theta0).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, log_r, _ = self.regressor(theta0_tensor, x_tensor)

        if theta1 is not None:
            theta1_tensor = tensor(theta1).to(self.device, self.dtype)
            _, log_r_1, _ = self.regressor(theta1_tensor, x_tensor)
            log_r = log_r - log_r_1

        log_r = log_r.detach().numpy()

        if log:
            return log_r
        return np.exp(log_r)

    def predict_score(self, theta, x):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, x.shape[0])

        self.regressor = self.regressor.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, _, t_hat = self.regressor(theta_tensor, x_tensor)

        t_hat = t_hat.detach().numpy()

        return t_hat

