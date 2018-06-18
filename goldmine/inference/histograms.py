import numpy as np
import logging

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

            logging.info('Initialized histogram with the following settings:')
            logging.info('  Bins per parameter:  %s', self.n_bins_theta)
            logging.info('  Bins per observable: %s', self.n_bins_x)

            # Not yet trained
            self.n_parameters = None
            self.n_observables = None
            self.n_bins = None
            self.edges = None
            self.histo = None

        else:
            self.n_parameters = None
            self.n_observables = None
            self.edges = np.load(filename + '_edges.npy')
            self.histo = np.load(filename + '_histo.npy')
            self.n_bins = self.histo.shape

            logging.info('Loaded histogram:')
            logging.info('  Filename:            %s', filename + '_*.npy')
            logging.info('  Number of bins:      %s', self.n_bins)

            # TODO: change self.edges from numpy to list (for saving + loading)

    def _calculate_binning(self, theta, x):

        all_theta_x = np.hstack([theta, x]).T

        # Number of bins
        n_samples = x.shape[0]
        n_parameters = theta.shape[1]
        n_observables = x.shape[1]

        n_bins_per_theta = self.n_bins_theta
        if n_bins_per_theta == 'auto':
            total_bins = 10 + int(round(n_samples ** (1. / 3.), 0))
            logging.debug(total_bins)
            n_bins_per_theta = max(5, int(round(total_bins ** (1. / (n_parameters + n_observables)))))

        n_bins_per_x = self.n_bins_x
        if n_bins_per_x == 'auto':
            total_bins = 10 + int(round(n_samples ** (1. / 3.), 0))
            n_bins_per_x = max(5, int(round(total_bins ** (1. / (n_parameters + n_observables)))))

        all_n_bins = [n_bins_per_theta] * n_parameters + [n_bins_per_x] * n_observables

        # Find edges based on percentiles
        all_edges = []
        all_ranges = []

        for data, n_bins in zip(all_theta_x, all_n_bins):
            edges = np.percentile(data, np.linspace(0., 100., n_bins + 1))
            range_ = (np.nanmin(data) - 0.01, np.nanmax(data) + 0.01)
            edges[0], edges[-1] = range_

            all_edges.append(edges)
            all_ranges.append(range_)

        all_edges = np.array(all_edges)
        all_ranges = np.array(all_ranges)

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

        logging.info('Filling histogram with settings:')
        logging.info('  theta given: %s', theta is not None)
        logging.info('  x given:     %s', x is not None)
        logging.info('  y given:     %s', y is not None)
        logging.info('  r_xz given:  %s', r_xz is not None)
        logging.info('  t_xz given:  %s', t_xz is not None)
        logging.info('  Samples:     %s', n_samples)
        logging.info('  Parameters:  %s', self.n_parameters)
        logging.info('  Obserables:  %s', self.n_observables)

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

        # Calculate cell volumes
        original_shape = tuple(self.n_bins)
        flat_shape = tuple([-1] + list(self.n_bins[self.n_parameters:]))

        bin_widths = [this_edges[1:] - this_edges[:-1] for this_edges in self.edges[self.n_parameters:]]

        ################################################################################################################
        # TODO: Calculate nd volumes from the 1d widths

        # bin_widths: (n_observables, n_bins) -> 1d widths
        # self.n_bins: (n_parameters + n_observables,) -> n_bins

        # volume: (n_bins0, ..., n_binsn) -> 1.
        # Loop over all observable bins (i0 ... in)
        # volume[i0 ... in] = bin_widths[0, i0] * bin_widths[1, i1] * ... * bin_widths[n, in]

        volumes = np.ones(flat_shape[1:])
        # for i in range(self.n_parameters):
        #     volumes[] *= np.broadcast(bin_widths[i], flat_shape[1:])
        #  TODO: Figure out how to broadcast along a given axis
        ################################################################################################################

        # Normalize histograms (for each theta bin)
        histo = histo.reshape(flat_shape)

        for i in range(histo.shape[0]):
            histo[i] /= volumes
            histo[i] /= np.sum(histo[i])

        histo = histo.reshape(original_shape)
        self.histo = histo

    def save(self, filename):
        if self.histo is None:
            raise ValueError('Histogram has to be trained (filled) before being saved!')

        np.save(filename + '_edges.npy', self.edges)
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
