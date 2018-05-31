import autograd.numpy as np


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
