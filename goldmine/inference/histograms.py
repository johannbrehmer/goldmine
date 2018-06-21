import numpy as np
import logging
import pickle

from goldmine.inference.base import Inference


class HistogramInference(Inference):
    """ Base class for inference methods. """

    def __init__(self, **params):
        super().__init__()

        filename = params.get('filename', None)

        if filename is None:
            # Parameters for new MAF
            self.n_bins_theta = params.get('n_bins_theta', 'auto')
            self.n_bins_x = params.get('n_bins_x', 'auto')
            self.observables = params.get('observables', 'all')

            logging.info('Initialized histogram with the following settings:')
            logging.info('  Bins per parameter:  %s', self.n_bins_theta)
            logging.info('  Bins per observable: %s', self.n_bins_x)
            logging.info('  Observable indices:  %s', self.observables)

            # Not yet trained
            self.n_parameters = None
            self.n_observables = None
            self.n_bins = None
            self.edges = None
            self.histo = None

        else:
            self.n_parameters = None
            self.n_observables = None
            with open(filename + '_edges.pickle', 'rb') as file:
                self.edges = pickle.load(file)
            self.histo = np.load(filename + '_histo.npy')
            self.n_bins = self.histo.shape

            logging.info('Loaded histogram:')
            logging.info('  Filename:            %s', filename + '_*.npy')
            logging.info('  Number of bins:      %s', self.n_bins)

    def _calculate_binning(self, theta, x):

        all_theta_x = np.hstack([theta, x]).T

        # Number of bins
        n_samples = x.shape[0]
        n_parameters = theta.shape[1]
        n_all_observables = x.shape[1]

        # Observables to actually plot
        if self.observables == 'all':
            self.observables = list(range(n_all_observables))

        n_binned_observables = len(self.observables)

        # TODO: better automatic bin number determination
        recommended_n_bins = 10 + int(round(n_samples ** (1. / 3.), 0))
        logging.info('Recommended total number of bins: %s', recommended_n_bins)

        n_bins_per_theta = self.n_bins_theta
        if n_bins_per_theta == 'auto':
            n_bins_per_theta = max(3, int(round(recommended_n_bins ** (1. / (n_parameters + n_binned_observables)))))

        n_bins_per_x = self.n_bins_x
        if n_bins_per_x == 'auto':
            n_bins_per_x = max(3, int(round(recommended_n_bins ** (1. / (n_parameters + n_binned_observables)))))

        all_n_bins = [1 for i in range(n_all_observables)]
        for i in self.observables:
            all_n_bins[i] = n_bins_per_x
        all_n_bins = [n_bins_per_theta] * n_parameters + all_n_bins

        # Find edges based on percentiles
        all_edges = []
        all_ranges = []

        for i, (data, n_bins) in enumerate(zip(all_theta_x, all_n_bins)):
            edges = np.percentile(data, np.linspace(0., 100., n_bins + 1))
            range_ = (np.nanmin(data) - 0.01, np.nanmax(data) + 0.01)
            edges[0], edges[-1] = range_

            # Remove zero-width bins
            widths = np.array(list(edges[1:] - edges[:-1]) + [1.])
            edges = edges[widths > 1.e-9]

            all_n_bins[i] = len(edges) - 1
            all_edges.append(edges)
            all_ranges.append(range_)

        return all_n_bins, all_edges, all_ranges

    def requires_class_label(self):
        return False

    def requires_joint_ratio(self):
        return False

    def requires_joint_score(self):
        return False

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None,
            **params):

        n_samples = x.shape[0]
        self.n_parameters = theta.shape[1]
        self.n_observables = x.shape[1]
        fill_empty_bins = params.get('fill_empty_bins', False)

        logging.info('Filling histogram with settings:')
        logging.info('  theta given:   %s', theta is not None)
        logging.info('  x given:       %s', x is not None)
        logging.info('  y given:       %s', y is not None)
        logging.info('  r_xz given:    %s', r_xz is not None)
        logging.info('  t_xz given:    %s', t_xz is not None)
        logging.info('  Samples:       %s', n_samples)
        logging.info('  Parameters:    %s', self.n_parameters)
        logging.info('  Observables:    %s', self.n_observables)
        logging.info('  No empty bins: %s', fill_empty_bins)

        # Find bins
        logging.info('Calculating binning')
        self.n_bins, self.edges, ranges = self._calculate_binning(theta, x)

        logging.info('Bin edges:')
        for i, (this_bins, this_range, this_edges) in enumerate(zip(self.n_bins, ranges, self.edges)):
            if i < theta.shape[1]:
                logging.info(
                    '  theta %s: %s bins, range %s, edges %s',
                    i + 1, this_bins, this_range, this_edges
                )
            else:
                logging.info(
                    '  x %s:     %s bins, range %s, edges %s',
                    i + 1 - theta.shape[1], this_bins, this_range, this_edges
                )

        # Fill histograms
        logging.info('Filling histograms')
        theta_x = np.hstack([theta, x])

        histo, _ = np.histogramdd(theta_x, bins=self.edges, range=ranges, normed=False, weights=None)

        # Avoid empty bins
        if fill_empty_bins:
            histo[histo<=1.] = 1.

        # Calculate cell volumes
        original_shape = tuple(self.n_bins)
        flat_shape = tuple([-1] + list(self.n_bins[self.n_parameters:]))

        bin_widths = [this_edges[1:] - this_edges[:-1] for this_edges in self.edges[self.n_parameters:]]

        volumes = np.ones(flat_shape[1:])
        for obs in range(self.n_observables):
            # Broadcast bin widths to array with shape like volumes
            bin_widths_broadcasted = np.ones(flat_shape[1:])
            for indices in np.ndindex(flat_shape[1:]):
                bin_widths_broadcasted[indices] = bin_widths[obs][indices[obs]]
            volumes[:] *= bin_widths_broadcasted

        # Normalize histograms (for each theta bin)
        histo = histo.reshape(flat_shape)

        for i in range(histo.shape[0]):
            histo[i] /= volumes
            histo[i] = histo[i] / np.sum(histo[i] * volumes)

        histo = histo.reshape(original_shape)

        # Avoid NaNs
        histo[np.invert(np.isfinite(histo))] = 0.

        self.histo = histo

    def save(self, filename):
        if self.histo is None:
            raise ValueError('Histogram has to be trained (filled) before being saved!')

        with open(filename + '_edges.pickle', 'wb') as file:
            pickle.dump(self.edges, file)
        np.save(filename + '_histo.npy', self.histo)

    def predict_density(self, theta, x, log=False):
        theta_x = np.hstack([theta, x])

        all_indices = []

        for j in range(theta_x.shape[1]):
            indices = np.searchsorted(self.edges[j],
                                      theta_x[:, j],
                                      side="right") - 1

            indices[indices < 0] = 0
            indices[indices >= self.n_bins[j]] = self.n_bins[j] - 1

            # logging.debug('Obs %s, bins %s, indices %s', j,self.n_bins[j], indices)

            all_indices.append(indices)

        if log:
            return np.log(self.histo[all_indices])
        return self.histo[all_indices]

    def predict_ratio(self, theta0, theta1, x):
        raise NotImplementedError()

    def predict_score(self, theta, x):
        raise NotImplementedError()

    def generate_samples(self, theta):
        raise NotImplementedError()
