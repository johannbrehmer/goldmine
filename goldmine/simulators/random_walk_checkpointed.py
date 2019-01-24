import autograd.numpy as np
import autograd as ag
from itertools import product
import logging

from goldmine.simulators.base import CheckpointedSimulator, SimulationTooLongException
from goldmine.various.utils import check_random_state


class CheckpointedRandomWalk(CheckpointedSimulator):
    """
    Simulator for a Lotka-Volterra predator-prey scenario.

    Setup follows appendix F of https://arxiv.org/pdf/1605.06376.pdf very closely. Note, however, that as parameters
    theta we use the log of the parameters of that paper! In other words, the parameters multiplying the predator and
    prey numbers in the differential equation are the components of exp(theta).
    """

    def __init__(self, initial_state=0, duration=1., delta_t=0.1):

        super().__init__()

        # Save parameters
        self.initial_state = 0
        self.duration = duration
        self.delta_t = delta_t
        self.n_time_series = int(self.duration / self.delta_t) + 1

        # Parameters
        self.n_parameters = 1

        # Autograd
        self._d_simulate_step = ag.grad_and_aux(self._simulate_step)

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Parameters
        tmin, tmax = 10., 10000.
        theta0_default = 100.
        theta1_value = 100.

        # Single benchmark point
        if single_theta:
            return (np.array([theta0_default]).reshape((1, -1)),
                    np.array([theta1_value]).reshape((1, -1)))

        # Ranges
        if random:
            benchmarks = np.random.rand(n_thetas, self.n_parameters)
        else:
            benchmarks = np.linspace(0., 1., n_thetas)

        # Rescale and exponentiate to correct ranges
        benchmarks[:] *= (np.log(tmax) - np.log(tmin))
        benchmarks[:] += np.log(tmin)
        benchmarks = np.exp(benchmarks)

        theta1 = theta1_value * np.ones_like(benchmarks)

        return benchmarks, theta1

    def theta_grid_default(self, n_points_per_dim=5):

        return self.theta_defaults(n_thetas=n_points_per_dim, random=False)

    def _simulate_step(self, theta_score,
                       start_state, start_time, next_recorded_time, start_steps,
                       rng, theta=None, thetas_additional=None,
                       max_steps=100000, steps_warning=10000):

        # Thetas for the evaluation of the likelihood (ratio / score)
        if theta is None:
            theta = np.copy(theta_score)

        thetas_eval = [theta_score]
        if thetas_additional is not None:
            thetas_eval += thetas_additional
        n_eval = len(thetas_eval)

        # Initial state
        state = np.copy(start_state)
        simulated_time = start_time
        n_steps = start_steps

        # Track log p(x, z)
        logp_xz = [0. for _ in thetas_eval]

        # Possible events
        event_effects = np.array([-1, 1], dtype=np.int)

        # Gillespie algorithm
        while next_recorded_time > simulated_time:

            # Rates of different possible events
            rates = np.array([100., theta[0]])
            total_rate = np.sum(rates)

            # Time of next event
            try:
                interaction_time = rng.exponential(scale=1. / total_rate)
            except ValueError:  # Raised when done in autograd mode for score
                interaction_time = rng.exponential(scale=1. / total_rate._value)

            # Choose next event
            event = 0 if rng.rand(1) < rates[0] / total_rate else 1

            logging.debug("Step %s at t = %s", "up" if event == 1 else "down", interaction_time)

            # Calculate and sum log probability
            for k in range(n_eval):
                rates_eval = np.array([100., thetas_eval[k][0]])
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

    def _simulate(self, theta_score, rng, theta=None, thetas_additional=None, max_steps=100000, steps_warning=10000, epsilon=1.e-9, extract_score=True):

        # Output
        t_xz_steps = []
        logp_xz_steps = []
        time_series = np.zeros(self.n_time_series, dtype=np.int)

        # Initial state
        state = np.array([self.initial_state], dtype=np.int)
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
                )
                t_xz_step = None

            # Save state, joint ratio, score
            time_series[i] = int(state)

            if extract_score:
                t_xz_steps.append(t_xz_step)
            logp_xz_steps.append(logp_xz_step)

            # Prepare for next step
            next_recorded_time += self.delta_t

        if extract_score:
            t_xz_steps = np.array(t_xz_steps)
        else:
            t_xz_steps = None

        if logp_xz_steps is None or None in logp_xz_steps:
            logp_xz_steps = None
        else:
            logp_xz_steps = np.array(logp_xz_steps)  # (checkpoints, thetas)
        time_series = time_series.astype(np.int)

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
                if logp_xz_checkpoints is not None:
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
                t_xz = np.sum(t_xz_checkpoints, axis=0)
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
        x = time_series[:].astype(np.float)

        # Mean of time series
        mean_x = np.mean(x)

        # Variance of time series
        var_x = np.var(x, ddof=1)

        # Normalize for correlation coefficients
        x_norm = (x - mean_x) / np.sqrt(var_x)

        # auto correlation coefficient
        autocorr_x = []
        for lag in [1, 2]:
            autocorr_x.append(np.dot(x_norm[:-lag], x_norm[lag:]) / (n - 1))

        summary_statistics = [mean_x, np.log(var_x + 1)] + autocorr_x
        return np.array(summary_statistics)

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
