import logging

from goldmine.inference.base import CheckpointedInference
from goldmine.ml.models.score import CheckpointScoreEstimator
from goldmine.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from goldmine.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from goldmine.ml.trainer import train_model
from goldmine.ml.losses import negative_log_likelihood, score_mse
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
            self.checkpoint_score_model = CheckpointScoreEstimator(
                n_parameters=n_parameters,
                n_latent=n_latent,
                n_hidden=tuple([n_step_units_per_layer] * n_step_hidden_layers),
                activation = step_activation
            )

            # Global model
            if n_components is not None and n_components > 1:
                self.maf = ConditionalMixtureMaskedAutoregressiveFlow(
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
                self.maf = ConditionalMaskedAutoregressiveFlow(
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
            raise NotImplementedError



    def requires_class_label(self):
        return False

    def requires_joint_ratio(self):
        return False

    def requires_joint_score(self):
        return False

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None, theta1=None,
            z_checkpoints=None, r_xz_checkpoints=None, t_xz_checkpoints=None,
            batch_size=64, initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
            early_stopping=True, **params):


        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def predict_density(self, theta, x):
        raise NotImplementedError()

    def predict_ratio(self, theta0, theta1, x):
        raise NotImplementedError()

    def predict_score(self, theta, x):
        raise NotImplementedError()

    def generate_samples(self, theta):
        raise NotImplementedError()
