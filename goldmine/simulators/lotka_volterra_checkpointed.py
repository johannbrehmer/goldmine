import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import CheckpointedSimulator, SimulationTooLongException
from goldmine.various.utils import check_random_state


class CheckpointedLotkaVolterra(CheckpointedSimulator):
    """
    Simulator for a Lotka-Volterra predator-prey scenario.

    Setup follows appendix F of https://arxiv.org/pdf/1605.06376.pdf very closely. Note, however, that as parameters
    theta we use the log of the parameters of that paper! In other words, the parameters multiplying the predator and
    prey numbers in the differential equation are the components of exp(theta).
    """

    def __init__(self, initial_predators=50, initial_prey=100, duration=30., delta_t=0.2,
                 use_summary_statistics=True, normalize_summary_statistics=True, use_full_time_series=False,
                 smear_summary_statistics=False, zoom_in=True):

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
        self.smear_summary_statistics = smear_summary_statistics
        self.zoom_in = zoom_in

        # Parameters
        self.n_parameters = 4

        # Autograd
        self._d_simulate_step = ag.grad_and_aux(self._simulate_step)

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            # return (np.log(np.array([0.01, 0.5, 1.0, 0.01])).reshape((1, -1)),
            #        np.array([-4.6, -0.5, 0., -4.6]).reshape((1, -1)))
            if self.zoom_in:
                return (np.log(np.array([0.01, 0.5, 1.0, 0.01])).reshape((1, -1)),
                        np.array([-4.61, -0.69, 0.00, -4.61]).reshape((1, -1)))
            return (np.log(np.array([0.01, 0.5, 1.0, 0.01])).reshape((1, -1)),
                    np.array([-4.6, -0.5, 0., -4.6]).reshape((1, -1)))

        # Ranges
        # theta_min = np.array([-5., -5., -5., -5.])
        # theta_max = np.array([2., 2., 2., 2.])
        theta_min = np.array([-5., -0.8, -0.3, -5.])
        theta_max = np.array([-4.5, -0.3, 0.2, -4.5])

        if self.zoom_in:
            theta_min = np.array([-4.62, -0.70, -0.01, -4.62])
            theta_max = np.array([-4.60, -0.68, 0.01, -4.60])

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

        theta1 = np.zeros_like(benchmarks)
        if self.zoom_in:
            theta1[:] = np.array([-4.61, -0.69, 0.00, -4.61])
        else:
            theta1[:] = np.array([-4.5, -0.5, 0., -4.5])

        return benchmarks, theta1

    def theta_grid_default(self, n_points_per_dim=5):
        # Default
        if n_points_per_dim is None or n_points_per_dim <= 0:
            n_points_per_dim = 5

        theta_min = np.array([-5., -0.8, -0.3, -5.]).reshape((4, 1))
        theta_max = np.array([-4.5, -0.3, 0.2, -4.5]).reshape((4, 1))
        if self.zoom_in:
            theta_min = np.array([-4.62, -0.70, -0.01, -4.62]).reshape((4, 1))
            theta_max = np.array([-4.60, -0.68, 0.01, -4.60]).reshape((4, 1))

        u = np.linspace(0., 1., n_points_per_dim).reshape((1, n_points_per_dim))

        grid = theta_min + u * (theta_max - theta_min)

        return grid

    def _simulate_step(self, theta_score,
                       start_state, start_time, next_recorded_time, start_steps,
                       rng, theta=None, thetas_additional=None,
                       max_steps=100000, steps_warning=10000,
                       epsilon=1.e-9):

        # Thetas for the evaluation of the likelihood (ratio / score)
        if theta is None:
            theta = np.copy(theta_score)

        thetas_eval = [theta_score]
        if thetas_additional is not None:
            thetas_eval += thetas_additional
        n_eval = len(thetas_eval)

        exp_theta = np.exp(theta)
        exp_thetas_eval = [np.exp(theta_eval) for theta_eval in thetas_eval]

        # Initial state
        state = np.copy(start_state)
        simulated_time = start_time
        n_steps = start_steps

        # Track log p(x, z)
        logp_xz = [0. for _ in exp_thetas_eval]

        # Possible events
        event_effects = np.array([
            [1, 0],  # Predator born
            [-1, 0],  # Predator dies
            [0, 1],  # Prey born
            [0, -1]  # Predator eats prey
        ], dtype=np.int)

        # Gillespie algorithm
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

            # Choose next event
            event = -1
            while event < 0:
                dice = rng.rand(1)
                for j in range(4):
                    if dice < np.sum(rates[:j + 1]) / total_rate:
                        event = j
                        break

            # Calculate and sum log probability
            for k in range(n_eval):
                rates_eval = np.array([
                    exp_thetas_eval[k][0] * state[0] * state[1],  # Predator born
                    exp_thetas_eval[k][1] * state[0],  # Predator dies
                    exp_thetas_eval[k][2] * state[1],  # Prey born
                    exp_thetas_eval[k][3] * state[0] * state[1]  # Predator eats prey
                ])
                total_rate_eval = np.sum(rates_eval)

                logp_xz[k] += (np.log(total_rate_eval) - interaction_time * total_rate_eval
                               + np.log(rates_eval[event]) - np.log(total_rate_eval))

            # Resolve event
            simulated_time += interaction_time
            state += event_effects[event]
            n_steps += 1

            # Handling long simulations
            if (n_steps + 1) % steps_warning == 0:
                logging.debug('Simulation is exceeding %s steps, simulated time: %s', n_steps, simulated_time)

            if n_steps > max_steps:
                logging.warning('Too many steps in simulation. Total rate: %s', total_rate)
                raise SimulationTooLongException()

        # Return state and everything else
        return logp_xz[0], (logp_xz[1:], state, simulated_time, n_steps)

    def _simulate(self, theta_score, rng, theta=None, thetas_additional=None, max_steps=100000, steps_warning=10000,
                  epsilon=1.e-9, extract_score=True):

        # Output
        t_xz_steps = []
        logp_xz_steps = []
        time_series = np.zeros([self.n_time_series, 2], dtype=np.int)

        # Initial state
        state = np.array([self.initial_predators, self.initial_prey], dtype=np.int)
        next_recorded_time = 0.
        simulated_time = epsilon
        n_steps = 0

        # Gillespie algorithm
        for i in range(self.n_time_series):
            # Run step
            if extract_score:
                t_xz_step, (logp_xz_step, state, simulated_time, n_steps) = self._d_simulate_step(
                    theta_score,
                    start_state=state,
                    start_time=simulated_time,
                    next_recorded_time=next_recorded_time,
                    start_steps=n_steps,
                    theta=theta,
                    rng=rng,
                    thetas_additional=thetas_additional,
                    max_steps=max_steps,
                    steps_warning=steps_warning,
                    epsilon=epsilon
                )
            else:
                _, (logp_xz_step, state, simulated_time, n_steps) = self._simulate_step(
                    theta_score,
                    start_state=state,
                    start_time=simulated_time,
                    next_recorded_time=next_recorded_time,
                    start_steps=n_steps,
                    theta=theta,
                    rng=rng,
                    thetas_additional=thetas_additional,
                    max_steps=max_steps,
                    steps_warning=steps_warning,
                    epsilon=epsilon
                )
                t_xz_step = None

            # Save state, joint ratio, score
            time_series[i,0] = int(state[0])
            time_series[i,1] = int(state[1])

            if extract_score:
                t_xz_steps.append(t_xz_step)
            logp_xz_steps.append(t_xz_step)

            # Prepare for next step
            next_recorded_time += self.delta_t

        if extract_score:
            t_xz_steps = np.array(t_xz_steps)
        else:
            t_xz_steps = None
        logp_xz_steps = np.array(logp_xz_steps)

        return time_series, t_xz_steps, logp_xz_steps

    def _simulate_until_success(self, theta, rng, thetas_additional=None, max_steps=100000, steps_warning=10000,
                                epsilon=1.e-9, max_tries=5):

        time_series = None
        logp_xz_checkpoints = None
        logp_xz = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            tries += 1
            try:
                time_series, _, logp_xz_checkpoints = self._simulate(
                    theta_score=theta,
                    rng=rng,
                    theta=theta,
                    thetas_additional=thetas_additional,
                    max_steps=max_steps,
                    steps_warning=steps_warning,
                    epsilon=epsilon,
                    extract_score=False
                )

                # Sum over checkpoints
                logp_xz_checkpoints = np.array(logp_xz_checkpoints)  # (checkpoints, thetas)
                logp_xz = np.sum(logp_xz_checkpoints, axis=0)
            except SimulationTooLongException:
                pass
            else:
                if time_series is None:  # This should not happen
                    raise RuntimeError('No time series result and no exception -- this should not happen!')

        if time_series is None:
            raise SimulationTooLongException(
                'Simulation exceeded {} steps in {} consecutive trials'.format(max_steps, max_tries))

        return logp_xz, time_series, logp_xz_checkpoints

    def _d_simulate_until_success(self, theta, rng, theta_score=None, thetas_additional=None, max_steps=100000,
                                  steps_warning=10000, epsilon=1.e-9, max_tries=5):
        if theta_score is None:
            theta_score = theta

        time_series = None
        t_xz = None
        logp_xz = None
        t_xz_checkpoints = None
        logp_xz_checkpoints = None
        tries = 0

        while time_series is None and (max_tries is None or max_tries <= 0 or tries < max_tries):
            tries += 1
            try:
                time_series, t_xz_checkpoints, logp_xz_checkpoints = self._simulate(
                    theta_score,
                    rng,
                    theta,
                    thetas_additional,
                    max_steps,
                    steps_warning,
                    epsilon,
                    extract_score=True
                )

                logging.debug(t_xz_checkpoints)

                # Sum over checkpoints
                logp_xz_checkpoints = np.array(logp_xz_checkpoints)  # (checkpoints, thetas)
                t_xz_checkpoints = np.array(t_xz_checkpoints)  # (checkpoints, parameters)
                logp_xz = np.sum(logp_xz_checkpoints, axis=0)
                t_xz = np.sum(logp_xz_checkpoints, axis=0)
            except SimulationTooLongException:
                pass
            else:
                if time_series is None:  # This should not happen
                    raise RuntimeError('No time series result and no exception -- this should not happen!')

        if time_series is None:
            raise SimulationTooLongException(
                'Simulation exceeded {} steps in {} consecutive trials'.format(max_steps, max_tries))

        # Only want steps
        return logp_xz, t_xz, time_series, logp_xz_checkpoints, t_xz_checkpoints

    def _calculate_observables(self, time_series, rng):
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

                # # Based on big prior
                # means = [7.05559643e+02, 3.97849297e+01, 7.34776178e+00, 4.51226290e+00,
                #          8.33611704e-01, 7.38606619e-01, 1.38464173e-01, 7.72252462e-02,
                #          6.52340705e-02]
                # stds = [2.90599684e+03, 5.31219626e+02, 3.54734035e+00, 1.31554388e+00,
                #         1.88679522e-01, 2.54926902e-01, 2.71919076e-01, 2.00932294e-01,
                #         3.55916090e-01]

                # Based on 'focus' prior
                means = [1.04272841e+02, 7.92735828e+01, 8.56355494e+00, 8.11906932e+00,
                         9.75067266e-01, 9.23352650e-01, 9.71107191e-01, 9.11167340e-01,
                         4.36308022e-02]
                stds = [2.68008281e+01, 2.14120703e+02, 9.00247450e-01, 1.04245882e+00,
                        1.13785497e-02, 2.63556410e-02, 1.36672075e-02, 2.76435894e-02,
                        1.38785995e-01]

                for i, (summary_statistic, mean, std) in enumerate(zip(summary_statistics, means, stds)):
                    summary_statistics[i] = (summary_statistic - mean) / std

            # Smear summary statistics
            if self.smear_summary_statistics:
                for i, summary_statistic in enumerate(zip(summary_statistics)):
                    noise = 0.05 * rng.rand(len(summary_statistic))
                    summary_statistics[i] += noise

            observables += summary_statistics

        # Full time series
        if self.use_full_time_series:
            observables += list(x)
            observables += list(y)

        observables = np.array(observables)

        return observables

    def rvs(self, theta, n, random_state=None, return_histories=False):
        logging.debug('Simulating %s evolutions for theta = %s', n, theta)

        rng = check_random_state(random_state)

        all_x = []
        histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

            _, time_series, _ = self._simulate_until_success(theta, rng)
            if return_histories:
                histories.append(time_series)

            x = self._calculate_observables(time_series, rng=rng)
            all_x.append(x)

        all_x = np.asarray(all_x)

        if return_histories:
            return all_x, histories
        return all_x

    def rvs_score(self, theta, theta_score, n, random_state=None):
        logging.debug('Simulating %s evolutions for theta = %s, augmenting with joint score', n, theta)

        rng = check_random_state(random_state, use_autograd=True)

        all_x = []
        all_t_xz = []
        all_t_xz_checkpoints = []
        all_histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

            _, t_xz, time_series, _, t_xz_checkpoints = self._d_simulate_until_success(theta, rng, theta_score)
            all_t_xz.append(t_xz)
            all_t_xz_checkpoints.append(t_xz_checkpoints)
            all_histories.append(time_series)

            x = self._calculate_observables(time_series, rng=rng)
            all_x.append(x)

        all_x = np.asarray(all_x)
        all_t_xz = np.asarray(all_t_xz)
        all_t_xz_checkpoints = np.asarray(all_t_xz_checkpoints)
        all_histories = np.asarray(all_histories)

        return all_x, all_t_xz, all_histories, all_t_xz_checkpoints

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):
        logging.debug('Simulating %s evolutions for theta = %s, augmenting with joint ratio between %s and %s',
                      n, theta, theta0, theta1)

        rng = check_random_state(random_state, use_autograd=True)

        all_x = []
        all_log_r_xz = []
        all_log_r_xz_checkpoints = []
        all_histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

            log_p_xzs, time_series, log_p_xz_checkpoints = self._simulate_until_success(theta, rng, [theta0, theta1])

            all_log_r_xz.append(log_p_xzs[0] - log_p_xzs[1])
            all_log_r_xz_checkpoints.append(log_p_xz_checkpoints[:, 0] - log_p_xz_checkpoints[:, 1])
            all_histories.append(time_series)

            x = self._calculate_observables(time_series, rng=rng)
            all_x.append(x)

        all_x = np.asarray(all_x)
        all_r_xz = np.exp(np.asarray(all_log_r_xz))
        all_r_xz_checkpoints = np.exp(np.asarray(all_log_r_xz_checkpoints))
        all_histories = np.asarray(all_histories)

        return all_x, all_r_xz, all_histories, all_r_xz_checkpoints

    def rvs_ratio_score(self, theta, theta0, theta1, theta_score, n, random_state=None):
        logging.debug('Simulating %s evolutions for theta = %s, augmenting with joint ratio between %s and %s and joint'
                      ' score at  %s', n, theta, theta0, theta1, theta_score)

        rng = check_random_state(random_state, use_autograd=True)

        all_x = []
        all_log_r_xz = []
        all_t_xz = []
        all_t_xz_checkpoints = []
        all_log_r_xz_checkpoints = []
        all_histories = []

        for i in range(n):
            logging.debug('  Starting sample %s of %s', i + 1, n)

            results = self._d_simulate_until_success(theta, rng, theta_score, [theta0, theta1])
            log_p_xzs, t_xz, time_series, log_p_xz_checkpoints, t_xz_checkpoints = results

            all_t_xz.append(t_xz)
            all_log_r_xz.append(log_p_xzs[0] - log_p_xzs[1])
            all_t_xz_checkpoints.append(t_xz_checkpoints)
            all_log_r_xz_checkpoints.append(log_p_xz_checkpoints[:, 0] - log_p_xz_checkpoints[:, 1])
            all_histories.append(time_series)

            x = self._calculate_observables(time_series, rng=rng)
            all_x.append(x)

        all_x = np.asarray(all_x)
        all_t_xz = np.asarray(all_t_xz)
        all_r_xz = np.exp(np.asarray(all_log_r_xz))
        all_r_xz_checkpoints = np.exp(np.asarray(all_log_r_xz_checkpoints))
        all_t_xz_checkpoints = np.asarray(all_t_xz_checkpoints)
        all_histories = np.asarray(all_histories)

        return all_x, all_r_xz, all_t_xz, all_histories, all_r_xz_checkpoints, all_t_xz_checkpoints
