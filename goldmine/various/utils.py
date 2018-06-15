import numpy as np
import logging
import sklearn


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
