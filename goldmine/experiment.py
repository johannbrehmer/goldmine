import numpy as np

from .various.utils import create_simulator


def run_simulator(simulator_name, theta0, theta1, n_samples_per_theta,
                  draw_from=None, generate_augmented_data=True,
                  folder='', filename_prefix='',
                  random_state=None):
    """
    Draws sample from a simulator.

    :param simulator_name: Specifies the simulator. Currently supported are 'galton' and 'epidemiology'.
    :param theta0: ndarray that provides a list of theta0 values (the numerator of the likelihood ratio as well as the
                   score reference point)
    :param theta1: ndarray that provides a list of theta1 values (the denominator of the likelihood ratio). Has to have
                   same shape as theta0.
    :param n_samples_per_theta: Number of samples per combination of theta0 and theta1.
    :param draw_from: list, either [0], [1], or None (= [0,1]). Determines whether theta0, theta1, or both are used for
                      the sampling.
    :param generate_augmented_data: bool, whether to ask the simulator for the  joint ratio and joint score.
    :param filename_prefix:
    :param folder:
    :param random_state: Numpy random state.
    """

    simulator = create_simulator(simulator_name)

    if draw_from is None:
        draw_from = [0, 1]
    if draw_from not in [[0], [1], [0, 1]]:
        raise ValueError('draw_from has value other than [0], [1], [0,1]: %s', draw_from)

    n_samples_per_theta_and_draw = n_samples_per_theta // len(draw_from)

    if generate_augmented_data:

        all_theta0 = []
        all_theta1 = []
        all_x = []
        all_y = []
        all_r_xz = []
        all_t_xz = []

        for theta0_, theta1_ in zip(theta0, theta1):
            for y in draw_from:
                x, r_xz, t_xy = simulator.rvs_ratio_score(
                    theta=theta0_,
                    theta0=theta0_,
                    theta1=theta1_,
                    theta_score=theta0_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )

                all_theta0 += [theta0_] * n_samples_per_theta_and_draw
                all_theta1 += [theta1_] * n_samples_per_theta_and_draw
                all_x += list(x)
                all_y += [y] * n_samples_per_theta_and_draw
                all_r_xz += list(all_r_xz)
                all_t_xz += list(all_t_xz)

        all_theta0 = np.array(all_theta0)
        all_theta1 = np.array(all_theta1)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_r_xz = np.array(all_r_xz)
        all_t_xz = np.array(all_t_xz)

        np.save(folder + '/' + filename_prefix + '_theta0' + '.npy', all_theta0)
        np.save(folder + '/' + filename_prefix + '_theta1' + '.npy', all_theta1)
        np.save(folder + '/' + filename_prefix + '_x' + '.npy', all_x)
        np.save(folder + '/' + filename_prefix + '_y' + '.npy', all_y)
        np.save(folder + '/' + filename_prefix + '_r_xz' + '.npy', all_r_xz)
        np.save(folder + '/' + filename_prefix + '_t_xz' + '.npy', all_t_xz)

    else:

        all_theta0 = []
        all_theta1 = []
        all_x = []
        all_y = []

        for theta0_, theta1_ in zip(theta0, theta1):
            for y in draw_from:
                x = simulator.rvs(
                    theta=theta0_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )

                all_theta0 += [theta0_] * n_samples_per_theta_and_draw
                all_theta1 += [theta1_] * n_samples_per_theta_and_draw
                all_x += list(x)
                all_y += [y] * n_samples_per_theta_and_draw

        all_theta0 = np.array(all_theta0)
        all_theta1 = np.array(all_theta1)
        all_x = np.array(all_x)
        all_y = np.array(all_y)

        np.save(folder + '/' + filename_prefix + '_theta0' + '.npy', all_theta0)
        np.save(folder + '/' + filename_prefix + '_theta1' + '.npy', all_theta1)
        np.save(folder + '/' + filename_prefix + '_x' + '.npy', all_x)
        np.save(folder + '/' + filename_prefix + '_y' + '.npy', all_y)
