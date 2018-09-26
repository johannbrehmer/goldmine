import logging
import numpy as np

from goldmine.inference.nde import MAFInference
from goldmine.ml.trainer import train_model
from goldmine.ml.losses import negative_log_likelihood, score_mse
from goldmine.ml.control_variates import t_xz_cv_corrected


class SCANDALInferenceWithRatioControlVariate(MAFInference):
    """ Neural conditional density estimation with masked autoregressive flows. """

    def requires_joint_score(self):
        return True

    def requires_joint_ratio(self):
        return True

    def fit(self,
            theta=None,
            x=None,
            y=None,
            r_xz=None,
            t_xz=None,
            theta1=None,
            batch_size=64,
            trainer='adam',
            initial_learning_rate=0.001,
            final_learning_rate=0.0001,
            n_epochs=50,
            validation_split=0.2,
            early_stopping=True,
            alpha=0.01,
            learning_curve_folder=None,
            learning_curve_filename=None,
            **params):
        """ Trains SCANDAL """

        logging.info('Training SCANDAL (MAF + score) with ratio control variate with settings:')
        logging.info('  alpha:          %s', alpha)
        logging.info('  theta given:    %s', theta is not None)
        logging.info('  x given:        %s', x is not None)
        logging.info('  y given:        %s', y is not None)
        logging.info('  r_xz given:     %s', r_xz is not None)
        logging.info('  t_xz given:     %s', t_xz is not None)
        logging.info('  theta1 given:   %s', theta1 is not None)
        logging.info('  Samples:        %s', x.shape[0])
        logging.info('  Parameters:     %s', theta.shape[1])
        logging.info('  Obserables:     %s', x.shape[1])
        logging.info('  Batch size:     %s', batch_size)
        logging.info('  Optimizer:      %s', trainer)
        logging.info('  Learning rate:  %s initially, decaying to %s', initial_learning_rate, final_learning_rate)
        logging.info('  Valid. split:   %s', validation_split)
        logging.info('  Early stopping: %s', early_stopping)
        logging.info('  Epochs:         %s', n_epochs)

        assert theta is not None
        assert x is not None
        assert t_xz is not None
        assert r_xz is not None
        assert theta1 is not None

        # First, calculate the coefficient for the control variate
        cov_r_t = []
        var_r = None

        for i in range(t_xz.shape[1]):
            covariance_matrix = np.cov(1. / r_xz, t_xz[:, i])
            var_r = covariance_matrix[0, 0]
            cov_r_t.append(covariance_matrix[0, 1])

        cov_r_t = np.array(cov_r_t)
        cv_coefficients = - cov_r_t / var_r

        logging.info('Calculating control variate coefficients')
        logging.info('  Variance in 1/r(x,z):                  %s', var_r)
        for i, cov_r_t_i in enumerate(cov_r_t):
            logging.info('  Covariance between t(x,z)_' + str(i) + ', 1/r(x,z): %s', cov_r_t_i)
        for i, cv_ci in enumerate(cv_coefficients):
            logging.info('  CV coefficient for t(x,z)_' + str(i) + ':           %s', cv_ci)

        train_model(
            model=self.maf,
            loss_functions=[negative_log_likelihood, score_mse],
            loss_weights=[1., alpha],
            loss_labels=['nll', 'score'],
            thetas=theta,
            theta1=theta1,
            xs=x,
            t_xzs=t_xz,
            pre_loss_transformer=lambda *args: t_xz_cv_corrected(*args, coefficients=cv_coefficients),
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
