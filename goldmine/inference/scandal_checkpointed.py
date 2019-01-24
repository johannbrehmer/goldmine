import logging
import numpy as np
import torch
from torch import tensor

from goldmine.inference.base import CheckpointedInference
from goldmine.ml.models.score import CheckpointScoreEstimator
from goldmine.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from goldmine.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from goldmine.ml.models.checkpoint import FlowCheckpointScoreModel
from goldmine.ml.trainer_checkpoint import train_checkpointed_model
from goldmine.ml.losses_checkpoint import negative_log_likelihood, score_mse, score_checkpoint_mse
from goldmine.various.utils import expand_array_2d


class CheckpointedSCANDALInference(CheckpointedInference):
    """ Neural conditional density estimation with masked autoregressive flows. """

    def __init__(self, **params):
        super().__init__()

        filename = params.get('filename', None)

        if filename is None:

            # Parameters
            n_parameters = params['n_parameters']
            n_observables = params['n_observables']
            n_latent = params['n_latent']

            n_components = params.get('n_components', 1)
            n_mades = params.get('n_mades', 2)
            n_made_hidden_layers = params.get('n_made_hidden_layers', 2)
            n_made_units_per_layer = params.get('n_made_units_per_layer', 20)
            activation = params.get('activation', 'relu')
            batch_norm = params.get('batch_norm', False)

            n_step_hidden_layers = params.get('n_step_hidden_layers', 2)
            n_step_units_per_layer = params.get('n_step_units_per_layer', 20)
            step_activation = params.get('step_activation', 'relu')

            logging.info('Initialized checkpointed NDE (MAF) with the following settings:')
            logging.info('  Parameters:        %s', n_parameters)
            logging.info('  Observables:       %s', n_observables)
            logging.info('  Latent vars:       %s', n_latent)
            logging.info('  Checkpoint score estimator:')
            logging.info('    Hidden layers:   %s', n_step_hidden_layers)
            logging.info('    Units:           %s', n_step_units_per_layer)
            logging.info('    Activation:      %s', step_activation)
            logging.info('  Global flow:')
            logging.info('    Base components: %s', n_components)
            logging.info('    MADEs:           %s', n_mades)
            logging.info('    Hidden layers:   %s', n_made_hidden_layers)
            logging.info('    Units:           %s', n_made_units_per_layer)
            logging.info('    Activation:      %s', activation)
            logging.info('    Batch norm:      %s', batch_norm)

            # Step model
            checkpoint_score_model = CheckpointScoreEstimator(
                n_parameters=n_parameters,
                n_latent=n_latent,
                n_hidden=tuple([n_step_units_per_layer] * n_step_hidden_layers),
                activation=step_activation
            )

            # Global model
            if n_components is not None and n_components > 1:
                maf = ConditionalMixtureMaskedAutoregressiveFlow(
                    n_components=n_components,
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_hiddens=tuple([n_made_units_per_layer] * n_made_hidden_layers),
                    n_mades=n_mades,
                    activation=activation,
                    batch_norm=batch_norm,
                    input_order='random',
                    mode='sequential',
                    alpha=0.1
                )
            else:
                maf = ConditionalMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_hiddens=tuple([n_made_units_per_layer] * n_made_hidden_layers),
                    n_mades=n_mades,
                    activation=activation,
                    batch_norm=batch_norm,
                    input_order='random',
                    mode='sequential',
                    alpha=0.1
                )

            # Wrapper
            self.model = FlowCheckpointScoreModel(maf, checkpoint_score_model)

        else:
            self.model = torch.load(filename + '.pt', map_location='cpu')

            logging.info('Loaded checkpointed SCANDAL from file:')
            logging.info('  Filename:        %s', filename)
            logging.info('  Checkpoint score estimator:')
            logging.info('    Hidden layers: %s', self.step_model.n_hiddens)
            logging.info('    Activation:    %s', self.step_model.activation)
            logging.info('    Parameters:    %s', self.step_model.n_parameters)
            logging.info('    Latents:       %s', self.step_model.n_latent)
            logging.info('  Global flow:')
            logging.info('    Parameters:    %s', self.model.global_model.n_conditionals)
            logging.info('    Observables:   %s', self.model.global_model.n_inputs)
            try:
                logging.info('    Components:    %s', self.model.global_model.n_components)
            except AttributeError:
                logging.info('    Components:    1')
            logging.info('    MADEs:         %s', self.model.global_model.n_mades)
            logging.info('    Hidden layers: %s', self.model.global_model.n_hiddens)
            logging.info('    Activation:    %s', self.model.global_model.activation)
            logging.info('    Batch norm:    %s', self.model.global_model.batch_norm)

        # Have everything on CPU (unless training)
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def requires_class_label(self):
        return False

    def requires_joint_ratio(self):
        return False

    def requires_joint_score(self):
        return True

    def fit(
            self,
            theta=None,
            x=None,
            y=None,
            r_xz=None,
            t_xz=None,
            theta1=None,
            z_checkpoints=None,
            r_xz_checkpoints=None,
            t_xz_checkpoints=None,
            batch_size=64,
            trainer='adam',
            initial_learning_rate=0.001,
            final_learning_rate=0.0001,
            n_epochs=50,
            validation_split=0.2,
            early_stopping=True,
            alpha=1.,
            beta=1.,
            learning_curve_folder=None,
            learning_curve_filename=None,
            **params
    ):
        """ Trains checkpointed flow """

        logging.info('Training checkpointed SCANDAL with settings:')
        logging.info('  alpha:                  %s', alpha)
        logging.info('  beta:                   %s', beta)
        logging.info('  theta given:            %s', theta is not None)
        logging.info('  theta1 given:           %s', theta1 is not None)
        logging.info('  x given:                %s', x is not None)
        logging.info('  y given:                %s', y is not None)
        logging.info('  r_xz given:             %s', r_xz is not None)
        logging.info('  t_xz given:             %s', t_xz is not None)
        logging.info('  z_checkpoints given:    %s', z_checkpoints is not None)
        logging.info('  r_xz_checkpoints given: %s', r_xz_checkpoints is not None)
        logging.info('  t_xz_checkpoints given: %s', t_xz_checkpoints is not None)
        logging.info('  Samples:                %s', x.shape[0])
        logging.info('  Parameters:             %s', theta.shape[1])
        logging.info('  Obserables:             %s', x.shape[1])
        logging.info('  Checkpoints:            %s', z_checkpoints.shape[1])
        logging.info('  Latent variables:       %s', z_checkpoints.shape[2])
        logging.info('  Batch size:             %s', batch_size)
        logging.info('  Optimizer:              %s', trainer)
        logging.info('  Learning rate:          %s initially, decaying to %s', initial_learning_rate,
                     final_learning_rate)
        logging.info('  Valid. split:           %s', validation_split)
        logging.info('  Early stopping:         %s', early_stopping)
        logging.info('  Epochs:                 %s', n_epochs)

        train_checkpointed_model(
            model=self.model,
            loss_functions=[negative_log_likelihood, score_mse, score_checkpoint_mse],
            loss_weights=[1., alpha, beta],
            loss_labels=['nll', 'score', 'checkpoint_score'],
            thetas=theta,
            xs=x,
            ys=None,
            r_xz_checkpoints=r_xz_checkpoints,
            t_xz_checkpoints=t_xz_checkpoints,
            z_checkpoints=z_checkpoints,
            batch_size=batch_size,
            trainer=trainer,
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
            n_epochs=n_epochs,
            validation_split=validation_split,
            early_stopping=early_stopping,
            learning_curve_folder=learning_curve_folder,
            learning_curve_filename=learning_curve_filename
        )

    def save(self, filename):
        # Fix a bug in pyTorch, see https://github.com/pytorch/text/issues/350
        self.model.to()
        torch.save(self.model, filename + '.pt')
        self.model.to(self.device, self.dtype)

    def predict_score_conditional_on_checkpoints(self, theta, z_checkpoints):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, z_checkpoints.shape[0])

        self.model = self.model.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        z_checkpoints = tensor(z_checkpoints).to(self.device, self.dtype)

        t_checkpoints = self.model.forward_checkpoints(theta, z_checkpoints)
        t_checkpoints = torch.sum(t_checkpoints, dim=1)
        t_checkpoints = t_checkpoints.detach().numpy()

        return t_checkpoints

    def predict_density(self, theta, x, log=False):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, x.shape[0])

        self.model = self.model.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, log_likelihood = self.model.global_model.log_likelihood(theta_tensor, x_tensor)
        log_likelihood = log_likelihood.detach().numpy()

        if log:
            return log_likelihood
        return np.exp(log_likelihood)

    def predict_ratio(self, theta0, theta1, x, log=False):
        # If just one theta given, broadcast to number of samples
        theta0 = expand_array_2d(theta0, x.shape[0])
        theta1 = expand_array_2d(theta1, x.shape[0])

        self.model = self.model.to(self.device, self.dtype)
        theta0_tensor = tensor(theta0).to(self.device, self.dtype)
        theta1_tensor = tensor(theta1).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, log_likelihood_theta0 = self.model.global_model.log_likelihood(theta0_tensor, x_tensor)
        _, log_likelihood_theta1 = self.model.global_model.log_likelihood(theta1_tensor, x_tensor)

        log_likelihood_theta0 = log_likelihood_theta0.detach().numpy()
        log_likelihood_theta1 = log_likelihood_theta1.detach().numpy()

        if log:
            return log_likelihood_theta0 - log_likelihood_theta1
        return np.exp(log_likelihood_theta0 - log_likelihood_theta1)

    def predict_score(self, theta, x):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, x.shape[0])

        self.model = self.model.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, _, score = self.model.global_model.log_likelihood_and_score(theta_tensor, x_tensor)

        score = score.detach().numpy()

        return score

    def generate_samples(self, theta):
        self.model = self.model.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)

        samples = self.model.global_model.generate_samples(theta_tensor).detach().numpy()
        return samples
