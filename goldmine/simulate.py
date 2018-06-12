#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_simulator
    from goldmine.various.utils import general_init
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    print(sys.path)
    from goldmine.various.look_up import create_simulator
    from goldmine.various.utils import general_init


def simulate(simulator_name,
             sample_label,
             theta0=None,
             theta1=None,
             draw_from=None,
             grid_sampling=False,
             generate_joint_ratio=True,
             generate_joint_score=True,
             n_samples_per_theta=1000,
             random_state=None):
    """
    Draws sample from a simulator.

    :param simulator_name: Specifies the simulator. Currently supported are 'galton' and 'epidemiology'.
    :param theta0: None or ndarray that provides a list of theta0 values (the numerator of the likelihood ratio as well
                   as the score reference point). If None, load simulator defaults.
    :param theta1: None or ndarray that provides a list of theta1 values (the denominator of the likelihood ratio) with
                   same shape as theta0. If None, load simulator defaults.
    :param draw_from: list, either [0], [1], or None (= [0,1]). Determines whether theta0, theta1, or both are used for
                      the sampling.
    :param generate_joint_ratio: bool, whether to ask the simulator for the joint ratio (only if theta1 is given).
    :param generate_joint_score: bool, whether to ask the simulator for the joint score.
    :param n_samples_per_theta: Number of samples per combination of theta0 and theta1.
    :param filename_prefix:
    :param folder:
    :param random_state: Numpy random state.
    """

    logging.info('Starting simulation')
    logging.info('  Simulator:            %s', simulator_name)
    logging.info('  Sample:               %s', sample_label)
    logging.info('  theta0:               %s', 'default' if theta0 is None else theta0)
    logging.info('  theta1:               %s', 'default' if theta1 is None else theta1)
    if theta0 is None:
        logging.info('  theta sampling:       %s', 'grid' if grid_sampling else 'random')
    logging.info('  Samples / theta:      %s', n_samples_per_theta)
    logging.info('  Generate joint ratio: %s', generate_joint_ratio)
    logging.info('  Generate joint score: %s', generate_joint_score)

    simulator = create_simulator(simulator_name)

    # Load data
    if theta0 is not None:
        theta0 = np.load(base_dir + '/goldmine/data/thetas/' + simulator_name + '/' + theta0)

    if theta1 is not None:
        theta1 = np.load(base_dir + '/goldmine/data/thetas/' + simulator_name + '/' + theta1)

    # Filenames
    folder = base_dir + '/goldmine/data/samples/' + simulator_name
    filename = simulator_name + '_' + sample_label

    # Default thetas
    if theta0 is None:
        theta0, theta1 = simulator.theta_defaults(random=not grid_sampling)

    # Check thetas
    has_theta1 = (theta1 is not None)

    if has_theta1:
        if theta1.shape != theta0.shape:
            raise ValueError('theta0 and theta1 have different shapes: %s, %s', theta0.shape, theta1.shape)
        if draw_from is None:
            draw_from = [0, 1]
        if draw_from not in [[0], [1], [0, 1]]:
            raise ValueError('draw_from has value other than [0], [1], [0,1]: %s', draw_from)

    else:
        theta1 = np.empty_like(theta0)
        theta1[:] = np.NaN
        generate_joint_ratio = False

        if draw_from is None:
            draw_from = [0]
        if draw_from not in [[0]]:
            raise ValueError('No theta1, and draw_from has value other than [0]: %s', draw_from)

    n_samples_per_theta_and_draw = n_samples_per_theta // len(draw_from)

    # Data to be generated
    all_theta0 = []
    all_theta1 = []
    all_x = []
    all_y = []
    all_r_xz = []
    all_t_xz = []

    logging.info('Parameter points:')
    logging.info('theta0 = %s', theta0)
    if has_theta1:
        logging.info('theta1 = %s', theta1)

    # Loop over thetas and run simulator
    for theta0_, theta1_ in zip(theta0, theta1):
        for y in draw_from:

            if generate_joint_ratio and generate_joint_score:
                x, r_xz, t_xy = simulator.rvs_ratio_score(
                    theta=theta0_,
                    theta0=theta0_,
                    theta1=theta1_,
                    theta_score=theta0_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )
            elif generate_joint_ratio:
                x, r_xz = simulator.rvs_ratio(
                    theta=theta0_,
                    theta0=theta0_,
                    theta1=theta1_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )
            elif generate_joint_score:
                x, r_xz, t_xy = simulator.rvs_score(
                    theta=theta0_,
                    theta_score=theta0_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )
            else:
                x = simulator.rvs(
                    theta=theta0_,
                    n=n_samples_per_theta_and_draw,
                    random_state=random_state
                )

            all_theta0 += [theta0_] * n_samples_per_theta_and_draw
            all_theta1 += [theta1_] * n_samples_per_theta_and_draw
            all_x += list(x)
            all_y += [y] * n_samples_per_theta_and_draw
            if generate_joint_ratio:
                all_r_xz += list(all_r_xz)
            if generate_joint_score:
                all_t_xz += list(all_t_xz)

    logging.info('Saving results')

    # Save results
    np.save(folder + '/' + filename + '_theta0' + '.npy', all_theta0)
    np.save(folder + '/' + filename + '_theta1' + '.npy', all_theta1)
    np.save(folder + '/' + filename + '_x' + '.npy', all_x)
    np.save(folder + '/' + filename + '_y' + '.npy', all_y)
    if generate_joint_ratio:
        np.save(folder + '/' + filename + '_r_xz' + '.npy', all_r_xz)
    if generate_joint_score:
        np.save(folder + '/' + filename + '_t_xz' + '.npy', all_t_xz)


def main():
    """ Starts simulation """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator', help='Simulator: "galton" or "epidemiology"')
    parser.add_argument('sample', help='Sample name ("train" or "test")')
    parser.add_argument('--theta0', default=None, help='Theta0 file, defaults to standard parameters')
    parser.add_argument('--theta1', default=None, help='Theta1 file, defaults to no theta1')
    parser.add_argument('--gridsampling', action='store_true', help='If argument theta0 is not set, samples theta0 on a'
                                                                    + ' grid rather than randomly')
    parser.add_argument('--nsamples', default=100, help='Number of samples per theta value')
    parser.add_argument('--noratio', action='store_true', help='Do not generate joint ratio')
    parser.add_argument('--noscore', action='store_true', help='Do not generate joint score')

    args = parser.parse_args()

    # Start simulation
    simulate(
        args.simulator,
        args.sample,
        theta0=args.theta0,
        theta1=args.theta1,
        n_samples_per_theta=args.nsamples,
        generate_joint_ratio=not args.noratio,
        generate_joint_score=not args.noscore,
        grid_sampling=args.gridsampling
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
