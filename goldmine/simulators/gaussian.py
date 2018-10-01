import numpy as np
from scipy.stats import norm

from goldmine.simulators.base import Simulator


class GaussianSimulator(Simulator):
    """ Simple simulator example drawing samples from a conditional Gaussian """

    def get_discretization(self):
        return None,

    def __init__(self, width=0.5):
        super(GaussianSimulator, self).__init__()

        # Parameters
        self.n_parameters = 1
        self.width = width

    def theta_defaults(self, n_thetas=100, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.array([1.])], None

        # Ranges
        theta_min = np.array([0.])
        theta_max = np.array([2.])

        # Generate benchmarks in [0,1]^n_parameters
        if random:
            benchmarks = np.random.rand(n_thetas).reshape((-1, 1))

        else:
            benchmarks = np.linspace(0., 2., n_thetas).reshape((-1, 1))

        # Rescale to correct ranges
        benchmarks[:] *= (theta_max - theta_min)
        benchmarks[:] += theta_min

        return benchmarks.reshape((-1, 1)), None

    def rvs(self, theta, n, random_state=None):

        x0 = np.hstack((
            norm.rvs(theta, self.width, n // 4),
            norm.rvs(0., self.width, n // 4),
            norm.rvs(0., self.width, n // 4),
            norm.rvs(-theta, self.width, n // 4)
        ))
        x1 = np.hstack((
            norm.rvs(0., self.width, n // 4),
            norm.rvs(-theta, self.width, n // 4),
            norm.rvs(theta, self.width, n // 4),
            norm.rvs(0., self.width, n // 4)
        ))

        x = np.vstack((x0, x1)).T

        return x

    def rvs_score(self, theta, theta_score, n, random_state=None):

        x = self.rvs(theta, n)

        x_a = x[:n // 4, :]
        t_a = (x_a[:, 0] - theta) / self.width ** 2

        x_b = x[n // 4:2 * (n // 4), :]
        t_b = -(x_b[:, 1] + theta) / self.width ** 2

        x_c = x[2 * (n // 4):3 * (n // 4), :]
        t_c = (x_c[:, 1] - theta) / self.width ** 2

        x_d = x[3 * (n // 4):4 * (n // 4), :]
        t_d = -(x_d[:, 0] + theta) / self.width ** 2

        t = np.concatenate((t_a, t_b, t_c, t_d), axis=0).reshape((-1, 1))

        return x, t
