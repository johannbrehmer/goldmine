class GoldSimulator():

    """ Base class for simulators with access to joint score and joint likelihood ratios. """

    def __init__(self):
        raise NotImplementedError()

    def simulator_name(self):
        raise NotImplementedError()

    def rvs(self, theta, n):
        raise NotImplementedError()

    def rvs_score(self, theta, n):
        raise NotImplementedError()

    def rvs_ratio(self, theta, theta1, n):
        raise NotImplementedError()

    def rvs_ratio_score(self, theta, theta1, n):
        raise NotImplementedError()
