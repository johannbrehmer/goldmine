import numpy as np
import torch
from torch import tensor

from goldmine.inference.base import Inference
from goldmine.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from goldmine.ml.trainers import train
from goldmine.ml.losses import negative_log_likelihood_loss


class MAFInference(Inference):
    """ Neural conditional density estimation with masked autoregressive flows. """

    def __init__(self,
                 n_parameters,
                 n_observables,
                 n_mades=3,
                 n_made_hidden_layers=3,
                 n_made_units_per_layer=20,
                 batch_norm=False):
        super().__init__()

        self.maf = ConditionalMaskedAutoregressiveFlow(
            n_conditionals=n_parameters,
            n_inputs=n_observables,
            n_hiddens=tuple([n_made_units_per_layer] * n_made_hidden_layers),
            n_mades=n_mades,
            batch_norm=batch_norm,
            input_order='sequential',
            mode='sequential',
            alpha=0.1
        )

    def requires_class_label(self):
        return False

    def requires_joint_ratio(self):
        return False

    def requires_joint_score(self):
        return False

    def predicts_density(self):
        return True

    def predicts_ratio(self):
        return True

    def predicts_score(self):
        return True

    def fit(self,
            theta=None,
            x=None,
            y=None,
            r_xz=None,
            t_xz=None,
            batch_size=64,
            initial_learning_rate=0.001,
            final_learning_rate=0.0001,
            n_epochs=50,
            learning_curve_folder=None,
            learning_curve_filename=None):
        """ Trains MAF """

        train(
            model=self.maf,
            loss_function=negative_log_likelihood_loss,
            thetas=theta,
            xs=x,
            ys=None,
            batch_size=batch_size,
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
            n_epochs=50,
            learning_curve_folder=learning_curve_folder,
            learning_curve_filename=learning_curve_filename
        )

    def save(self, filename):
        torch.save(self.maf.state_dict(), filename)

    def load(self, filename):
        self.maf.load_state_dict(torch.load(filename))

    def predict_density(self, x=None, theta=None, log=False):
        log_likelihood = self.maf.predict_log_likelihood(tensor(theta), tensor(x)).numpy()

        if log:
            return log_likelihood
        return np.exp(log_likelihood)

    def predict_ratio(self, x=None, theta=None, theta1=None, log=False):
        log_likelihood_theta0 = self.maf.predict_log_likelihood(tensor(theta), tensor(x)).numpy()
        log_likelihood_theta1 = self.maf.predict_log_likelihood(tensor(theta1), tensor(x)).numpy()

        if log:
            return log_likelihood_theta0 - log_likelihood_theta1
        return np.exp(log_likelihood_theta0 - log_likelihood_theta1)

    def predict_score(self, x=None, theta=None):
        score = self.maf.predict_score(tensor(theta), tensor(x)).numpy()

        return score
