import autograd.numpy as np
import autograd as ag
from itertools import product

from goldmine.simulators.base import Simulator


class Epidemiology(Simulator):
    """
    Simulator for a model of stropococcus transmission dynamics

    Model taken from E. Numminen et al., 'Estimating the Transmission Dynamics of Streptococcus pneumoniae from Strain
    Prevalence Data', Biometrics 69, 748-757 (2013).

    Settings (and their values in that paper):
    Number of strains: 53
    Number of individuals: 29
    Overall prevalence of strains: see Fig. I of said paper
    Number of time steps simulated: 10
    gamma: fixed to 1

    Parameters of interest and uniform priors in that paper:
    beta: 0...11
    Lambda: 0...2
    theta: 0...1

    Observables:

    """

    def __init__(self,
                 n_individuals=53,
                 n_strains=33,
                 overall_prevalence=None,
                 n_time_steps=10,
                 initial_infection=False,
                 use_original_summary_statistics=True,
                 use_prevalence_covariance=False):

        # Save parameters
        self.n_individuals = n_individuals
        self.n_strains = n_strains
        self.overall_prevalence = overall_prevalence
        self.initial_infection = initial_infection
        self.use_original_summary_statistics = use_original_summary_statistics
        self.use_prevalence_covariance = use_prevalence_covariance

        # Input
        if self.overall_prevalence is None:
            self.overall_prevalence = np.array(
                [0.12, 0.11, 0.10, 0.07, 0.06] + [0.05] * 3 + [0.04] * 2 + [0.03] * 3 + [0.02] * 3 + [0.015] * 7
                + [0.010] * 5 + [0.005] * 5
            )
        assert self.overall_prevalence.shape[0] == self.n_strains, 'Wrong number of strains in prevalence'

        # Parameters
        self.n_parameters = 3

    def theta_defaults(self, n_benchmarks=100, random=True):

        # Ranges
        theta_min = np.array([0., 0., 0.])
        theta_max = np.array([11., 2., 1.])

        # Generate benchmarks in [0,1]^n_parameters
        if random:
            benchmarks = np.random.rand(n_benchmarks, self.n_parameters)

        else:
            n_points_per_dimension = int(n_benchmarks ** (1 / self.n_parameters))

            if n_points_per_dimension ** self.n_parameters != n_benchmarks:
                raise Warning(
                    'Number of requested non-random parameter benchmarks not compatible with number of parameters.'
                    + ' Returning {0} benchmarks instead.'.format(n_points_per_dimension ** self.n_parameters)
                )

            benchmarks = np.array(
                list(
                    product(
                        *[np.linspace(0., 1., n_points_per_dimension) for i in range(self.n_parameters)]
                    )
                )
            )

        # Rescale to correct ranges
        benchmarks[:] += theta_min
        benchmarks[:] *= (theta_max - theta_min)

        return benchmarks

    def _draw_initial_state(self):

        if self.initial_infection:
            # Random numbers
            dice = np.rand(self.n_individuals, self.n_strains)

            # Infection threshold
            threshold = np.broadcast_to(self.overall_prevalence, (self.n_individuals, self.n_strains))

            # Initial infection states
            state = (dice < threshold)

        else:
            state = np.zeros((self.n_individuals, self.n_strains))

        return state

    def _time_step(self, old_state, theta):

        # Random numbers
        dice = np.rand(self.n_individuals, self.n_strains)

        # Exposure
        exposure = np.sum(
            old_state / (self.n_individuals - 1)
            * np.broadcast_to(1. / np.sum(old_state, axis=1), (self.n_individuals, self.n_strains)),
            axis=0
        )
        exposure = np.broadcast_to(exposure, (self.n_individuals, self.n_strains))

        # Prevalence
        prevalence = np.broadcast_to(self.overall_prevalence, (self.n_individuals, self.n_strains))

        # Individual infection status
        any_infection = (np.sum(old_state, axis=1) > 0)
        any_infection = np.broadcast_to(any_infection)

        # Infection threshold
        threshold = (
                np.invert(old_state)
                * (any_infection + np.invert(any_infection) * theta[2])
                * (theta[0] * exposure + theta[1] * prevalence)
        )

        # Initial infection states
        new_state = (dice < threshold)

        return new_state

    def _calculate_observables(self, state):
        pass

    def rvs(self, theta, n, random_state=None):

        x = []

        for i in range(n):
            # Initial infection
            state = self._draw_initial_state()

            # Spread
            for t in range(self.n_time_steps):
                state = self._time_step(state, theta)

            # Observables
            x.append(_calculate_observables(state))

        x = np.asarray(x)
        return x

    def rvs_score(self, theta, theta_score, n, random_state=None):
        pass

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        pass

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        pass
