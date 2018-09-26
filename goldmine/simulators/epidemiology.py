import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import Simulator
from goldmine.various.utils import check_random_state


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

    Parameters of interest and uniform priors in that paper:
    beta: 0...11
    Lambda: 0...2
    theta: 0...1
    gamma: fixed to 1

    Observables:

    """

    def __init__(self, n_individuals=53, n_strains=33, overall_prevalence=None, end_time=10, delta_t=0.1,
                 initial_infection=False, use_original_summary_statistics=True, use_prevalence_covariance=False):

        super().__init__()

        # Save parameters
        self.n_individuals = n_individuals
        self.n_strains = n_strains
        self.overall_prevalence = overall_prevalence
        self.end_time = end_time
        self.delta_t = delta_t
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
        self.n_parameters = 4

        # Autograd
        self._d_simulate_transmission = ag.grad_and_aux(self._simulate_transmission)

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.array([3.589, 0.593, 0.097, 1.])], None

        # Ranges
        # theta_min = np.array([0., 0., 0., 1.])
        # theta_max = np.array([11., 2., 1., 1.])
        theta_min = np.array([2, 0.3, 0.05, 0.5])
        theta_max = np.array([6., 1., 0.2, 2.])

        # Generate benchmarks in [0,1]^n_parameters
        if random:
            benchmarks = np.random.rand(n_thetas, self.n_parameters)

        else:
            n_free_parameters = 4
            n_points_per_dimension = int(n_thetas ** (1 / n_free_parameters))

            if n_points_per_dimension ** n_free_parameters != n_thetas:
                raise Warning(
                    'Number of requested grid parameter benchmarks not compatible with number of parameters.'
                    + ' Returning {0} benchmarks instead.'.format(n_points_per_dimension ** n_free_parameters)
                )

            benchmarks_free = list(
                product(
                    *[np.linspace(0., 1., n_points_per_dimension) for _ in range(n_free_parameters)]
                )
            )
            benchmarks = [[b[0], b[1], b[2], b[3]] for b in benchmarks_free]
            benchmarks = np.array(benchmarks)

        # Rescale to correct ranges
        benchmarks[:] *= (theta_max - theta_min)
        benchmarks[:] += theta_min

        return benchmarks, None

    def _simulate_transmission(self, theta, rng, return_history=False):

        # Track log p(x, z) (to calculate the score later)
        logp_xz = 0.

        # Initial state
        if self.initial_infection:
            dice = rng.rand(self.n_individuals, self.n_strains)
            threshold = np.broadcast_to(self.overall_prevalence, (self.n_individuals, self.n_strains))
            state = (dice < threshold)
        else:
            state = np.zeros((self.n_individuals, self.n_strains), dtype=np.bool)

        # Track state history
        if return_history:
            history = [state]

        # Time steps
        n_time_steps = int(round(self.end_time / self.delta_t))

        for i in range(n_time_steps):
            # Random numbers
            dice = rng.rand(self.n_individuals, self.n_strains)

            # Exposure
            exposure = (state / (self.n_individuals - 1.)
                        * np.broadcast_to(1. / np.sum(state, axis=1), (self.n_strains, self.n_individuals)).T)
            exposure[np.invert(np.isfinite(exposure))] = 0.
            exposure = np.sum(exposure, axis=0)
            exposure = np.broadcast_to(exposure, (self.n_individuals, self.n_strains))

            # Prevalence of each strain
            prevalence = np.broadcast_to(self.overall_prevalence, (self.n_individuals, self.n_strains))

            # Individual infection status
            any_infection = (np.sum(state, axis=1) > 0)
            any_infection = np.broadcast_to(any_infection, (self.n_strains, self.n_individuals)).T

            # Infection threshold
            probabilities_infected = (
                    np.invert(state)
                    * (any_infection * theta[2] + np.invert(any_infection))
                    * (theta[0] * exposure + theta[1] * prevalence)
                    * self.delta_t
                    + state * (1. - theta[3] * self.delta_t)
            )

            # Update state
            state = (dice < probabilities_infected)

            # Accumulate probabilities
            log_p_this_decision = state * probabilities_infected + (1 - state) * (1. - probabilities_infected)
            logp_xz = logp_xz + np.sum(np.log(log_p_this_decision))

            # Track state history
            if return_history:
                history.append(state)

        if return_history:
            return logp_xz, (state, history)

        return logp_xz, state

    def _calculate_observables(self, state):

        # Note that this is very different from the original paper, which uses an observation model tailored to the
        # collected data

        # Proportion of individuals with the most common strain (Numminem cross-check)
        strain_observations = np.sum(state, axis=0)
        most_common_strain = np.argmax(strain_observations)
        prevalence_most_common_strain = np.sum(state[:, most_common_strain], dtype=np.float) / float(self.n_individuals)

        # Number of singleton strains (= only on 1 individual, Numminem cross-check)
        n_singleton_strains = np.sum(strain_observations == 1)

        # Number of observed strains (Numminen 2)
        n_observed_strains = len(strain_observations[np.nonzero(strain_observations)])

        # Shannon entropy of observed strain distribution (Numminen 1)
        p_observed_strains = strain_observations[np.nonzero(strain_observations)]  # Remove zeros
        p_observed_strains = p_observed_strains.astype(np.float) / float(np.sum(p_observed_strains))  # Normalize
        shannon_entropy = - np.sum(p_observed_strains * np.log(p_observed_strains))

        # Any / multiple infections of individuals
        any_infection = (np.sum(state, axis=1) > 0)
        multiple_infections = (np.sum(state, axis=1) > 1)

        # Prevalence of any infection (Numminen 3)
        prevalence_any = np.sum(any_infection, dtype=np.float) / self.n_individuals

        # Prevalence of multiple infections (Numminen 4)
        prevalence_multiple = np.sum(multiple_infections, dtype=np.float) / self.n_individuals

        # Combine summary statistics
        summary_statistics = np.array([
            shannon_entropy,
            n_observed_strains,
            prevalence_any,
            prevalence_multiple,
            prevalence_most_common_strain,
            n_singleton_strains
        ])

        return summary_statistics

    def get_discretization(self):
        return None, 1, 1. / self.n_individuals, 1. / self.n_individuals, 1. / self.n_individuals, 1

    def rvs(self, theta, n, random_state=None, return_histories=False):

        logging.debug('Simulating %s epidemic evolutions for theta = %s', n, theta)

        rng = check_random_state(random_state)

        all_x = []
        histories = []

        for i in range(n):
            if return_histories:
                _, (state, history) = self._simulate_transmission(theta, rng, return_history=True)
                histories.append(history)
            else:
                _, state = self._simulate_transmission(theta, rng, return_history=False)

            x = self._calculate_observables(state)

            all_x.append(x)

        all_x = np.asarray(all_x)

        if return_histories:
            return all_x, histories
        return all_x

    def rvs_score(self, theta, theta_score, n, random_state=None, return_histories=False):

        logging.debug('Simulating %s epidemic evolutions for theta = %s, augmenting with joint score', n, theta)

        rng = check_random_state(random_state)

        all_x = []
        all_t_xz = []
        histories = []

        for i in range(n):
            if return_histories:
                t_xz, (state, history) = self._d_simulate_transmission(theta, rng, return_history=True)
                histories.append(history)
            else:
                t_xz, state = self._d_simulate_transmission(theta, rng, return_history=False)

            x = self._calculate_observables(state)

            all_x.append(x)
            all_t_xz.append(t_xz)

        all_x = np.asarray(all_x)
        all_t_xz = np.asarray(all_t_xz)

        if return_histories:
            return all_x, all_t_xz, histories
        return all_x, all_t_xz

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        pass

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        pass
