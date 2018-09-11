class SimulatorException(Exception):
    pass


class SimulationTooLongException(SimulatorException):
    pass


class Simulator:

    """ Base class for simulators with access to joint score and joint likelihood ratios. """

    def __init__(self):
        pass

    def get_discretization(self):
        raise NotImplementedError

    def theta_defaults(self, n_thetas=100, single_theta=False, random=True):
        raise NotImplementedError()

    def rvs(self, theta, n, random_state=None):
        raise NotImplementedError()

    def rvs_score(self, theta, theta_score, n, random_state=None):
        raise NotImplementedError()

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        raise NotImplementedError()

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        raise NotImplementedError()
