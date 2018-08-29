import autograd.numpy as np
import autograd as ag
from itertools import product

from goldmine.simulators.base import Simulator
from goldmine.simulators.epidemiology import Epidemiology


class Epidemiology2D(Epidemiology):
    """
    Simplified simulator for a model of stropococcus transmission dynamics with only 2 parameters of interest

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

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.n_parameters = 2

        # Autograd
        self._d_simulate_transmission = ag.grad_and_aux(self._simulate_transmission)

        # Two of the original four parameters are fixed
        self.fixed_lambda = 0.593
        self.fixed_gamma = 1.

    def theta_defaults(self, n_thetas=1000, single_theta=False, random=True):

        # Single benchmark point
        if single_theta:
            return [np.array([3.589, 0.097])], None

        # Ranges
        theta_min = np.array([2., 0.05])
        theta_max = np.array([6., 0.2])

        # Generate benchmarks in [0,1]^n_parameters
        if random:
            benchmarks = np.random.rand(n_thetas, self.n_parameters)

        else:
            n_points_per_dimension = int(n_thetas ** (1. / self.n_parameters))

            if n_points_per_dimension ** self.n_parameters != n_thetas:
                raise Warning(
                    'Number of requested grid parameter benchmarks not compatible with number of parameters.'
                    + ' Returning {0} benchmarks instead.'.format(n_points_per_dimension ** self.n_parameters)
                )

            benchmarks = list(
                product(
                    *[np.linspace(0., 1., n_points_per_dimension) for _ in range(self.n_parameters)]
                )
            )
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
                    * (any_infection * theta[1] + np.invert(any_infection))
                    * (theta[0] * exposure + self.fixed_lambda * prevalence)
                    * self.delta_t
                    + state * (1. - self.fixed_gamma * self.delta_t)
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

