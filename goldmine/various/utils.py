import numpy as np
import logging


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
