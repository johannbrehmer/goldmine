import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import Simulator, SimulationTooLongException
from goldmine.various.utils import check_random_state, get_size


class LotkaVolterra(Simulator):
    """
    Simulator for a Lotka-Volterra predator-prey scenario.

    Setup follows appendix F of https://arxiv.org/pdf/1605.06376.pdf very closely. Note, however, that as parameters
    theta we use the log of the parameters of that paper! In other words, the parameters multiplying the predator and
    prey numbers in the differential equation are the components of exp(theta).
    """

    def __init__(self, initial_predators=50, initial_prey=100, duration=30., delta_t=0.2,
                 use_summary_statistics=True, normalize_summary_statistics=True, use_full_time_series=False):

        super().__init__()

        # Save parameters
        self.initial_predators = initial_predators
        self.initial_prey = initial_prey
        self.duration = duration
        self.delta_t = delta_t
        self.n_time_series = int(self.duration / self.delta_t) + 1
        self.use_summary_statistics = use_summary_statistics
        self.normalize_summary_statistics = normalize_summary_statistics
        self.use_full_time_series = use_full_time_series

        # Parameters
        self.n_parameters = 4

        # Autograd
        self._d_simulate = ag.grad_and_aux(self._simulate)

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.log(np.array([0.01, 0.5, 1.0, 0.01]))], None

        # Ranges
        theta_min = np.array([-5., -5., -5., -5.])
        theta_max = np.array([2., 2., 2., 2.])

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
        benchmarks[:] *= (theta_max - theta_min)
        benchmarks[:] += theta_min

        benchmarks = benchmarks

        return benchmarks, None

    def theta_grid_default(self, n_points_per_dim=10):
        points_per_dim = np.linspace(-5, 2., n_points_per_dim)

        return [points_per_dim for _ in range(4)]

    def _simulate(self, theta, rng, max_steps=100000, steps_warning=10000, epsilon=1.e-9):

        # Exponentiated theta
        exp_theta = np.exp(theta)

        # Prepare recorded time series of states
        time_series = np.zeros([self.n_time_series, 2], dtype=np.int)

        # Initial state
        state = np.array([self.initial_predators, self.initial_prey], dtype=np.int)
        next_recorded_time = 0.
        simulated_time = 0.
        n_steps = 0

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
                    exp_theta[0] * state[0] * state[1],  # Predator born
                    exp_theta[1] * state[0],  # Predator dies
                    exp_theta[2] * state[1],  # Prey born
                    exp_theta[3] * state[0] * state[1]  # Predator eats prey
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

                if (n_steps + 1) % steps_warning == 0:
                    logging.debug('Simulation is exceeding %s steps, simulated time: %s', n_steps, simulated_time)

                if n_steps > max_steps:
                    logging.warning('Too many steps in simulation. Total rate: %s', total_rate)
                    raise SimulationTooLongException()

            time_series[i] = state.copy()
            next_recorded_time += self.delta_t

        return logp_xz, time_series

    def _simulate_until_success(self, theta, rng, max_steps=100000, steps_warning=10000, epsilon=1.e-9, max_tries=5):

        time_series = None
        logp_xz = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            tries += 1
            try:
                logp_xz, time_series = self._simulate(theta, rng, max_steps, steps_warning, epsilon)
            except SimulationTooLongException:
                pass
            else:
                if time_series is None:  # This should not happen
                    raise RuntimeError('No time series result and no exception -- this should not happen!')

        if time_series is None:
            raise SimulationTooLongException(
                'Simulation exceeded {} steps in {} consecutive trials'.format(max_steps, max_tries))

        return logp_xz, time_series

    def _d_simulate_until_success(self, theta, rng, max_steps=100000, steps_warning=10000, epsilon=1.e-9, max_tries=5):

        time_series = None
        t_xz = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            tries += 1
            try:
                t_xz, time_series = self._d_simulate(theta, rng, max_steps, steps_warning, epsilon)
            except SimulationTooLongException:
                pass
            else:
                if time_series is None:  # This should not happen
                    raise RuntimeError('No time series result and no exception -- this should not happen!')

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

            summary_statistics = [mean_x, mean_y, np.log(var_x + 1), np.log(var_y + 1)]
            summary_statistics += autocorr_x + autocorr_y + [cross_corr]

            # Normalize summary statistics
            if self.normalize_summary_statistics:
                # Normalize to mean expectation 0 and variance 1 (for prior), based on a pilot run
                means = [7.05559643e+02, 3.97849297e+01, 7.34776178e+00, 4.51226290e+00,
                         8.33611704e-01, 7.38606619e-01, 1.38464173e-01, 7.72252462e-02,
                         6.52340705e-02]
                stds = [2.90599684e+03, 5.31219626e+02, 3.54734035e+00, 1.31554388e+00,
                        1.88679522e-01, 2.54926902e-01, 2.71919076e-01, 2.00932294e-01,
                        3.55916090e-01]

                for i, (summary_statistic, mean, std) in enumerate(zip(summary_statistics, means, stds)):
                    summary_statistics[i] = (summary_statistic - mean) / std

            observables += summary_statistics

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

    def rvs(self, theta, n, random_state=None, return_histories=False):
        logging.debug('Simulating %s evolutions for theta = %s', n, theta)

        rng = check_random_state(random_state)

        all_x = []
        histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

            _, time_series = self._simulate_until_success(theta, rng)
            if return_histories:
                histories.append(time_series)

            x = self._calculate_observables(time_series)
            all_x.append(x)

        all_x = np.asarray(all_x)

        if return_histories:
            return all_x, histories
        return all_x

    def rvs_score(self, theta, theta_score, n, random_state=None, return_histories=False):
        logging.debug('Simulating %s evolutions for theta = %s, augmenting with joint score', n, theta)

        if np.linalg.norm(theta_score - theta) > 1.e-6:
            logging.error('Different values for theta and theta_score not yet supported!')
            raise NotImplementedError('Different values for theta and theta_score not yet supported!')

        rng = check_random_state(random_state, use_autograd=True)

        all_x = []
        all_t_xz = []
        histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

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
