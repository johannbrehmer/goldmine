import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import Simulator
from goldmine.various.utils import check_random_state


class LotkaVolterra(Simulator):
    """
    Simulator for a Lotka-Volterra predator-prey scenario.

    Setup follows appendix F of https://arxiv.org/pdf/1605.06376.pdf very closely.
    """

    def __init__(self, initial_predators=50, initial_prey=100, dutation=30., delta_t=0.2):

        super().__init__()

        # Save parameters
        self.initial_predators = initial_predators
        self.initial_prey = initial_prey
        self.duration = dutation
        self.delta_t = delta_t

        # Parameters
        self.n_parameters = 4

        # Autograd
        self._d_simulate = ag.grad_and_aux(self._simulate)

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.array([0.01, 0.5, 1.0, 0.01])], None

        # Ranges
        theta_min = np.exp(np.array([-5., -5., -5., -5.]))
        theta_max = np.exp(np.array([2., 2., 2., 2.]))

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

        # Rescale and exponentiate to correct ranges
        benchmarks[:] *= np.log(theta_max) - np.log(theta_min)
        benchmarks[:] += np.log(theta_min)

        benchmarks = np.exp(benchmarks)

        return benchmarks, None

    def _simulate(self, theta, rng):

        # Prepare recorded time series of states
        n_time_series = int(self.duration / self.dt) + 1
        time_series = np.zeros([n_time_series, 2], dtype=np.int)

        # Initial state
        state = np.array([self.initial_predators, self.initial_prey], dtype=np.int)
        next_recorded_time = 0.
        simulated_time = 0.

        # Track log p(x, z) (to calculate the score later)
        logp_xz = 0.

        # Possible events
        event_effects = np.array([
            [1, 0],  # Predator born
            [-1, 0],  # Predator dies
            [0, 1],  # Prey born
            [0, -1]  # Predator eats prey
        ], dtype=np.int)

        # Gillespie algorithm
        for i in range(n_time_series):

            while next_recorded_time > simulated_time:

                # Rates of different possible events
                rates = np.array([
                    theta[0] * state[0] * state[1],  # Predator born
                    theta[1] * state[0],  # Predator dies
                    theta[2] * state[1],  # Prey born
                    theta[3] * state[0] * state[1]  # Predator eats prey
                ])
                total_rate = np.sum(rates)

                # Time of next event
                interaction_time = rng.exponential(scale=1. / total_rate)
                simulated_time += interaction_time

                logp_xz += np.log(total_rate) - interaction_time * total_rate

                # Choose next event
                event = -1
                while event < 0:
                    dice = rng.random(1)
                    for j in range(4):
                        if dice < np.sum(rates[:j]) / total_rate:
                            event = j
                            break

                # Resolve next event
                state += event_effects[event]
                logp_xz += np.log(rates[event]) - np.log(total_rate)

            time_series[i] = state.copy()
            next_recorded_time += self.delta_t

        return logp_xz, time_series


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
    logging.info('Simulating %s epidemic evolutions for theta = %s', n, theta)

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
    logging.info('Simulating %s epidemic evolutions for theta = %s, augmenting with joint score', n, theta)

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
