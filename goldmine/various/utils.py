import numpy as np

from ..simulators.epidemiology import Epidemiology
from ..simulators.galton import Galton


def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-6):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def check_random_state(random_state):
    if random_state is None or isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return random_state


def create_simulator(simulator_name):
    if simulator_name == 'epidemiology':
        return Epidemiology()
    elif simulator_name == 'galton':
        return Galton()
    else:
        raise ValueError('Simulator name %s unknown'.format(simulator_name))


def create_inference(inference_name):
    raise ValueError('Inference technique name %s unknown'.format(inference_name))
