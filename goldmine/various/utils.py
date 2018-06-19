import numpy as np
import logging
import torch.nn.functional as F


def check_random_state(random_state):
    if random_state is None or isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return random_state


def general_init():
    logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG)

    logging.info('')
    logging.info('------------------------------------------------------------')
    logging.info('|                                                          |')
    logging.info('|  goldmine                                                |')
    logging.info('|                                                          |')
    logging.info('|              Experiments with simulator-based inference  |')
    logging.info('|                                                          |')
    logging.info('------------------------------------------------------------')
    logging.info('')

    logging.info('Hi! How are you today?')

    np.seterr(divide='ignore', invalid='ignore')


def shuffle(*arrays):
    """ Wrapper around sklearn.utils.shuffle that allows for Nones"""

    permutation = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        assert a.shape[0] == n_samples
        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def load_and_check(filename, warning_threshold=1.e9):
    data = np.load(filename)

    n_nans = np.sum(np.isnan(data))
    n_infs = np.sum(np.isinf(data))
    n_finite = np.sum(np.isfinite(data))

    if n_nans + n_infs > 0:
        logging.warning('Warning: file %s contains %s NaNs and %s Infs, compared to %s finite numbers!',
                        filename, n_nans, n_infs, n_finite)

    smallest = np.nanmin(data)
    largest = np.nanmax(data)

    if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
        logging.warning('Warning: file %s has some large numbers, rangin from %s to %s',
                        filename, smallest, largest)

    return data


def get_activation_function(activation_name):
    if activation_name == 'relu':
        return F.relu
    elif activation_name == 'tanh':
        return F.tanh
    elif activation_name == 'sigmoid':
        return F.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation_name)
