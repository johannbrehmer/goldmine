import numpy as np
import logging
import sklearn


def check_random_state(random_state):
    if random_state is None or isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return random_state


def general_init():
    logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG,
                        datefmt='%d.%m.%Y %H:%M:%S')
    logging.info('Hi! How are you today?')
    np.seterr(divide='ignore', invalid='ignore')


def shuffle(*arrays):
    """ Wrapper around sklearn.utils.shuffle that allows for Nones"""

    to_shuffle = []
    output_type = []

    for i, a in enumerate(arrays):
        if a is None:
            output_type.append((False, None))
        else:
            output_type.append((True, len(to_shuffle)))
            to_shuffle.append(a)

    shuffled = sklearn.utils.shuffle(to_shuffle)

    output = []
    for was_shuffled, index in output_type:
        if was_shuffled:
            output.append(shuffled[index])
        else:
            output.append(None)
    output = tuple(output)

    return output
