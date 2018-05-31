import autograd.numpy as np
import autograd as ag

from goldmine.simulators.base import Simulator
from goldmine.various.utils import sigmoid, check_random_state


class GeneralizedGaltonBoard(Simulator):
    """ Generalized Galton board example from arXiv:1805.XXXXX """

    def __init__(self, n_rows=20, n_nails=31):

        super().__init__()

        self.n_rows = n_rows
        self.n_nails = n_nails

        self.d_trace = ag.grad_and_aux(self.trace)  # for mining: calculate the gradient log_p_xz (the joint score)

    def nail_positions(self, theta, level=None, nail=None):
        if level is None or nail is None:
            level = np.broadcast_to(np.arange(self.n_rows), (self.n_nails, self.n_rows)).T
            nail = np.broadcast_to(np.arange(self.n_nails), (self.n_rows, self.n_nails))

        level_rel = 1. * level / (self.n_rows - 1)
        nail_rel = 2. * nail / (self.n_nails - 1) - 1.

        nail_positions = ((1. - np.sin(np.pi * level_rel)) * 0.5
                          + np.sin(np.pi * level_rel) * sigmoid(10 * theta * nail_rel))

        return nail_positions

    def threshold(self, theta, trace):
        begin, z = trace
        pos = begin
        level = 0
        for step in z:
            if step == 0:
                if level % 2 == 0:
                    pos = pos
                else:
                    pos = pos - 1
            else:
                if level % 2 == 0:
                    pos = pos + 1
                else:
                    pos = pos
            level += 1
        if level % 2 == 1:  # for odd rows, the first and last nails are constant
            if pos == 0:
                return 0.0
            elif pos == self.n_nails:
                return 1.0
        return self.nail_positions(theta, level, pos)

    def trace(self, theta, u, theta_ref=None):

        # Run and mine gold
        # left/right decisions are based on value of theta_ref (which defaults to theta if None)
        # but log_pxz based on value of theta (needed for ratio)

        if theta_ref is None:
            theta_ref = theta

        begin = pos = self.n_nails // 2
        z = []

        log_p_xz = 0.0

        while len(z) < self.n_rows:
            t_ref = self.threshold(theta_ref, (begin, z))
            t = self.threshold(theta, (begin, z))
            level = len(z)

            # going left
            if u[level] < t_ref or t_ref == 1.0:
                log_p_xz += np.log(t)

                if level % 2 == 0:  # even rows
                    pos = pos
                else:  # odd rows
                    pos = pos - 1

                z.append(0)

            # going right
            else:
                log_p_xz += np.log(1. - t)

                if level % 2 == 0:
                    pos = pos + 1
                else:
                    pos = pos

                z.append(1)

        x = pos

        return log_p_xz, x

    def simulator_name(self):
        return 'galton'

    def rvs(self,
            theta,
            n, random_state=None):

        """ Draws samples x according to p(x|theta) """

        rng = check_random_state(random_state)

        all_x = []

        for i in range(n):
            u = rng.rand(self.n_rows)
            _, x = self.trace(theta, u)
            all_x.append(x)

        return all_x

    def rvs_score(self,
                  theta, theta_score,
                  n, random_state=None):

        """ Draws samples x according to p(x|theta), augments them with joint score t(x, z | theta_score) """

        rng = check_random_state(random_state)

        all_x = []
        all_t_xz = []

        for i in range(n):
            u = rng.rand(self.n_rows)

            _, x = self.trace(theta_score, u, theta_ref=theta)
            t_xz, _ = self.d_trace(theta_score, u, theta_ref=theta)

            all_x.append(x)
            all_t_xz.append(t_xz)

        all_x = np.array(all_x)
        all_t_xz = np.array(all_t_xz)

        return all_x, all_t_xz

    def rvs_ratio(self, theta, theta0, theta1, n, random_state=None):

        """ Draws samples x according to p(x|theta), augments them with joint ratio r(x,z|theta0, theta1) """

        rng = check_random_state(random_state)

        all_x = []
        all_log_r_xz = []

        for i in range(n):
            u = rng.rand(self.n_rows)

            log_p_xz_theta0, x = self.trace(theta0, u, theta_ref=theta)
            log_p_xz_theta1, _ = self.trace(theta1, u, theta_ref=theta)

            all_x.append(x)
            all_log_r_xz.append(log_p_xz_theta0 - log_p_xz_theta1)

        all_x = np.array(all_x)
        all_r_xz = np.exp(np.array(all_log_r_xz))

        return all_x, all_r_xz

    def rvs_ratio_score(self,
                        theta, theta0, theta1, theta_score,
                        n, random_state=None):

        """ Draws samples x according to p(x|theta), augments them with joint ratio r(x,z|theta0,theta1) and
        joint score t(x,z|theta_score) """

        rng = check_random_state(random_state)

        all_x = []
        all_log_r_xz = []
        all_t_xz = []

        for i in range(n):
            u = rng.rand(self.n_rows)

            log_p_xz_theta0, x = self.trace(theta0, u, theta_ref=theta)
            log_p_xz_theta1, _ = self.trace(theta1, u, theta_ref=theta)
            t_xz, _ = self.d_trace(theta_score, u, theta_ref=theta)

            all_x.append(x)
            all_log_r_xz.append(log_p_xz_theta0 - log_p_xz_theta1)
            all_t_xz.append(t_xz)

        all_x = np.array(all_x)
        all_r_xz = np.exp(np.array(all_log_r_xz))
        all_t_xz = np.array(all_t_xz)

        return all_x, all_r_xz, all_t_xz
