import autograd.numpy as np
import autograd as ag

from goldmine.simulators.base import Simulator


class Epidemiology(Simulator):

    """
    Simulator for a model of stropococcus transmission dynamics

    Model taken from E. Numminen et al., 'Estimating the Transmission Dynamics of Streptococcus pneumoniae from Strain
    Prevalence Data', Biometrics 69, 748-757 (2013)
    """

    def __init__(self):
        raise NotImplementedError()

    def simulator_name(self):
        raise NotImplementedError()

    def rvs(self, theta, n, random_state=None):
        raise NotImplementedError()

    def rvs_score(self, theta, theta_score, n, random_state=None):
        raise NotImplementedError()

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        raise NotImplementedError()

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        raise NotImplementedError()
