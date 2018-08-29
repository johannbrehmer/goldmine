import numpy as np

from goldmine.simulators.base import Simulator


class GaussianSimulator(Simulator):
    """ Simple simulator example drawing samples from a conditional Gaussian """

    def get_discretization(self):
        return None,

    def __init__(self):
        super(GaussianSimulator, self).__init__()

        # Parameters
        self.n_parameters = 1

    def theta_defaults(self, n_thetas=100, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.array([0.])], None

        # Ranges
        theta_min = np.array([-1.])
        theta_max = np.array([1.])

        # Generate benchmarks in [0,1]^n_parameters
        if random:
            benchmarks = np.random.rand(n_thetas)

        else:
            benchmarks = np.linspace(0., 1., n_thetas).reshape((-1, 1))

        # Rescale to correct ranges
        benchmarks[:] *= (theta_max - theta_min)
        benchmarks[:] += theta_min

        benchmarks = benchmarks.reshape((-1, 1))

        return benchmarks, None

    def rvs(self, theta, n, random_state=None):

        x = np.random.normal(theta, 1. + theta ** 2, n).reshape((-1, 1))

        return x

    def rvs_score(self, theta, theta_score, n, random_state=None):

        x = np.random.normal(theta, 1. + theta ** 2, n).reshape((-1, 1))

        # log p = const - 0.5 * (x - theta)**2 / (1 + theta**2)
        # score = 0.5 * 2 * (x - theta) / (1 + theta**2) + 0.5 * (x - theta)**2 / (1 + theta**2)**2 * 2 * theta
        score = ((x - theta_score) / (1 + theta_score ** 2)
                 + theta_score * (x - theta_score) ** 2 / (1 + theta_score ** 2) ** 2)

        return x, score

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        raise NotImplementedError()

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        raise NotImplementedError()
