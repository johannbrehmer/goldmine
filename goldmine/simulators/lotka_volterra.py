import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import Simulator
from goldmine.various.utils import check_random_state


class SimulationTooLongException(Exception):

    def __str__(self):
        return 'Simulation exceeded the maximum number of steps'


class LotkaVolterra(Simulator):
    """
    Simulator for a Lotka-Volterra predator-prey scenario.

    Setup follows appendix F of https://arxiv.org/pdf/1605.06376.pdf very closely. One difference is that we do not
    normalize the summary statistics based on a sample run (since we do not calculate Euclidean distance between summary
    statistics, this is not absolutely necessary).
    """

    def __init__(self, initial_predators=50, initial_prey=100, duration=30., delta_t=0.2, use_summary_statistics=True,
                 use_full_time_series=False):

        super().__init__()

        # Save parameters
        self.initial_predators = initial_predators
        self.initial_prey = initial_prey
        self.duration = duration
        self.delta_t = delta_t
        self.n_time_series = int(self.duration / self.delta_t) + 1
        self.use_summary_statistics = use_summary_statistics
        self.use_full_time_series = use_full_time_series

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

    def _simulate(self, theta, rng, max_steps=10000000, epsilon=1.e-9):

        # Prepare recorded time series of states
        time_series = np.zeros([self.n_time_series, 2], dtype=np.int)

        # Initial state
        state = np.array([self.initial_predators, self.initial_prey], dtype=np.int)
        next_recorded_time = 0.
        simulated_time = 0.
        n_steps = 0.

        # Track log p(x, z) (to calculate the joint score with autograd)
        logp_xz = 0.

        # Possible events
        event_effects = np.array([
            [1, 0],  # Predator born
            [-1, 0],  # Predator dies
            [0, 1],  # Prey born
            [0, -1]  # Predator eats prey
        ], dtype=np.int)

        # Gillespie algorithm
        for i in range(self.n_time_series):

            while next_recorded_time > simulated_time:

                # Rates of different possible events
                rates = np.array([
                    theta[0] * state[0] * state[1],  # Predator born
                    theta[1] * state[0],  # Predator dies
                    theta[2] * state[1],  # Prey born
                    theta[3] * state[0] * state[1]  # Predator eats prey
                ])
                total_rate = np.sum(rates)

                if total_rate <= epsilon:  # Everyone is dead. Nothing will ever happen again.
                    simulated_time = self.duration + 1.
                    break

                # Time of next event
                try:
                    interaction_time = rng.exponential(scale=1. / total_rate)
                except ValueError:  # Raised when done in autograd mode for score
                    interaction_time = rng.exponential(scale=1. / total_rate._value)
                simulated_time += interaction_time

                logp_xz += np.log(total_rate) - interaction_time * total_rate

                # Choose next event
                event = -1
                while event < 0:
                    dice = rng.rand(1)
                    for j in range(4):
                        if dice < np.sum(rates[:j + 1]) / total_rate:
                            event = j
                            break

                # Resolve next event
                state += event_effects[event]
                logp_xz += np.log(rates[event]) - np.log(total_rate)

                # Count steps
                n_steps += 1

                if n_steps > max_steps:
                    logging.warning('Too many steps in simulation. Total rate: %s', total_rate)
                    raise SimulationTooLongException()

            time_series[i] = state.copy()
            next_recorded_time += self.delta_t

        return logp_xz, time_series

    def _simulate_until_success(self, theta, rng, max_steps=10000000, epsilon=1.e-9, max_tries=1000):

        time_series = None
        logp_xz = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            try:
                logp_xz, time_series = self._simulate(theta, rng, max_steps, epsilon)
            except SimulationTooLongException:
                tries += 1

        if time_series is None:
            raise SimulationTooLongException(
                'Simulation exceeded {} steps in {} consecutive trials'.format(max_steps, max_tries))

        return logp_xz, time_series

    def _d_simulate_until_success(self, theta, rng, max_steps=10000000, epsilon=1.e-9, max_tries=1000):

        time_series = None
        t_xz = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            try:
                t_xz, time_series = self._d_simulate(theta, rng, max_steps, epsilon)
            except SimulationTooLongException:
                tries += 1

        if time_series is None:
            raise SimulationTooLongException(
                'Simulation exceeded {} steps in {} consecutive trials'.format(max_steps, max_tries))

        return t_xz, time_series

    def _calculate_observables(self, time_series):
        """ Calculates observables: a combination of summary statistics and the full time series  """

        n = time_series.shape[0]
        x = time_series[:, 0].astype(np.float)
        y = time_series[:, 1].astype(np.float)

        observables = []

        # Calculate summary statistics, following 1605.06376
        if self.use_summary_statistics:

            # Mean of time series
            mean_x = np.mean(x)
            mean_y = np.mean(y)

            # Variance of time series
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)

            # Normalize for correlation coefficients
            x_norm = (x - mean_x) / np.sqrt(var_x)
            y_norm = (y - mean_y) / np.sqrt(var_y)

            # auto correlation coefficient
            autocorr_x = []
            autocorr_y = []
            for lag in [1, 2]:
                autocorr_x.append(np.dot(x_norm[:-lag], x_norm[lag:]) / (n - 1))
                autocorr_y.append(np.dot(y_norm[:-lag], y_norm[lag:]) / (n - 1))

            # cross correlation coefficient
            cross_corr = np.dot(x_norm, y_norm) / (n - 1)

            observables += [mean_x, mean_y, np.log(var_x + 1), np.log(var_y + 1)]
            observables += autocorr_x + autocorr_y + [cross_corr]

        # Full time series
        if self.use_full_time_series:
            observables += list(x)
            observables += list(y)

        observables = np.array(observables)

        return observables

    def get_discretization(self):
        discretization = []

        if self.use_summary_statistics:
            discretization += [1. / self.n_time_series, 1. / self.n_time_series, None, None, None, None, None, None,
                               None]

        if self.use_full_time_series:
            discretization += [1.] * 2 * self.n_time_series

        return tuple(discretization)

    def rvs(self, theta, n, random_state=None, return_histories=False, max_failures=5):
        logging.info('Simulating %s evolutions for theta = %s', n, theta)

        rng = check_random_state(random_state)

        all_x = []
        histories = []

        for i in range(n):

            logging.debug('Starting simulation %s / %s for theta = %s', i + 1, n, theta)

            _, time_series = self._simulate_until_success(theta, rng)
            if return_histories:
                histories.append(time_series)

            x = self._calculate_observables(time_series)
            all_x.append(x)

        all_x = np.asarray(all_x)

        if return_histories:
            return all_x, histories
        return all_x

    def rvs_score(self, theta, theta_score, n, random_state=None, return_histories=False, max_failures=5):
        logging.info('Simulating %s epidemic evolutions for theta = %s, augmenting with joint score', n, theta)

        if np.linalg.norm(theta_score - theta) > 1.e-6:
            logging.error('Different values for theta and theta_score not yet supported!')
            raise NotImplementedError('Different values for theta and theta_score not yet supported!')

        rng = check_random_state(random_state, use_autograd=True)

        all_x = []
        all_t_xz = []
        histories = []

        for i in range(n):

            t_xz, time_series = self._d_simulate_until_success(theta, rng)
            all_t_xz.append(t_xz)
            if return_histories:
                histories.append(time_series)

            x = self._calculate_observables(time_series)
            all_x.append(x)

        all_x = np.asarray(all_x)
        all_t_xz = np.asarray(all_t_xz)

        if return_histories:
            return all_x, all_t_xz, histories
        return all_x, all_t_xz
