#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_simulator
    from goldmine.various.utils import general_init, create_missing_folders, get_size
    from goldmine.simulators.base import SimulatorException
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    from goldmine.various.look_up import create_simulator
    from goldmine.various.utils import general_init, create_missing_folders, get_size
    from goldmine.simulators.base import SimulatorException


def simulate(simulator_name,
             sample_label,
             theta0=None,
             theta1=None,
             draw_from=None,
             single_theta=False,
             grid_sampling=False,
             generate_joint_ratio=True,
             generate_joint_score=True,
             checkpoint=False,
             n_thetas=1000,
             n_samples_per_theta=1000,
             random_state=None,
             continue_after_exceptions=True):
    """
    Draws sample from a simulator.

    :param continue_after_exceptions:
    :param single_theta:
    :param grid_sampling:
    :param sample_label:
    :param simulator_name: Specifies the simulator. Currently supported are 'galton' and 'epidemiology'.
    :param theta0: None or ndarray that provides a list of theta0 values (the numerator of the likelihood ratio as well
                   as the score reference point). If None, load simulator defaults.
    :param theta1: None or ndarray that provides a list of theta1 values (the denominator of the likelihood ratio) with
                   same shape as theta0. If None, load simulator defaults.
    :param draw_from: list, either [0], [1], or None (= [0,1]). Determines whether theta0, theta1, or both are used for
                      the sampling.
    :param generate_joint_ratio: bool, whether to ask the simulator for the joint ratio (only if theta1 is given).
    :param generate_joint_score: bool, whether to ask the simulator for the joint score.
    :param checkpoint: bool, whether to use a checkpointed version of the simulator.
    :param n_thetas: int, number of thetas samples of theta0 is None and single_theta is False
    :param n_samples_per_theta: Number of samples per combination of theta0 and theta1.
    :param random_state: Numpy random state.
    """

    logging.info('Starting simulation')
    logging.info('  Simulator:                 %s', simulator_name)
    logging.info('  Checkpoint:                %s', checkpoint)
    logging.info('  Sample:                    %s', sample_label)
    logging.info('  theta0:                    %s', 'default' if theta0 is None else theta0)
    logging.info('  theta1:                    %s', 'default' if theta1 is None else theta1)
    if theta0 is None:
        if single_theta:
            logging.info('  theta sampling:            single theta')
        else:
            logging.info('  theta sampling:            %s', ('grid' if grid_sampling else 'random'))
            logging.info('  Number of thetas:          %s', n_thetas)
    logging.info('  Samples / theta:           %s', n_samples_per_theta)
    logging.info('  Generate joint ratio:      %s', generate_joint_ratio)
    logging.info('  Generate joint score:      %s', generate_joint_score)
    logging.info('  Continue after exceptions: %s', continue_after_exceptions)

    # Check paths
    create_missing_folders(base_dir, simulator_name)

    simulator = create_simulator(simulator_name, checkpoint)

    # Load data
    if theta0 is not None:
        theta0 = np.load(base_dir + '/goldmine/data/thetas/' + simulator_name + '/' + theta0)

    if theta1 is not None:
        theta1 = np.load(base_dir + '/goldmine/data/thetas/' + simulator_name + '/' + theta1)

    # Filenames
    folder = base_dir + '/goldmine/data/samples/' + simulator_name
    filename = sample_label
    if single_theta:
        filename += '_singletheta'

    # Default thetas
    if theta0 is None:
        theta0, theta1 = simulator.theta_defaults(single_theta=single_theta,
                                                  n_thetas=n_thetas,
                                                  random=not grid_sampling)

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

        if generate_joint_ratio:
            logging.warning('Joint ratio requested, but theta1 not given -- will just generate joint score.')
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
    all_z_checkpoints = []
    all_r_xz_checkpoints = []
    all_t_xz_checkpoints = []

    logging.info('Parameter points:')
    logging.info('  theta0 = %s', theta0)
    if has_theta1:
        logging.info('  theta1 = %s', theta1)

    # Loop over thetas and run simulator
    n_simulations = len(list(zip(theta0, theta1)))
    n_verbose = max(n_simulations // 100, 1)

    for i_simulation, (theta0_, theta1_) in enumerate(zip(theta0, theta1)):

        if (i_simulation + 1) % n_verbose == 0:
            logging.info('Starting simulation for parameter setup %s / %s: theta0 = %s, theta1 = %s', i_simulation + 1,
                         n_simulations, theta0_, theta1_)

        for y in draw_from:

            t_xz = None
            r_xz = None
            z_checkpoints = None
            r_xz_checkpoints = None
            t_xz_checkpoints = None

            try:
                if checkpoint and generate_joint_ratio and generate_joint_score:
                    x, r_xz, t_xz, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints = simulator.rvs_ratio_score(
                        theta=theta0_,
                        theta0=theta0_,
                        theta1=theta1_,
                        theta_score=theta0_,
                        n=n_samples_per_theta_and_draw,
                        random_state=random_state
                    )
                elif checkpoint and generate_joint_ratio:
                    x, r_xz, z_checkpoints, r_xz_checkpoints = simulator.rvs_ratio(
                        theta=theta0_,
                        theta0=theta0_,
                        theta1=theta1_,
                        n=n_samples_per_theta_and_draw,
                        random_state=random_state
                    )
                elif checkpoint and generate_joint_score:
                    x, t_xz, z_checkpoints, t_xz_checkpoints = simulator.rvs_score(
                        theta=theta0_,
                        theta_score=theta0_,
                        n=n_samples_per_theta_and_draw,
                        random_state=random_state
                    )
                elif generate_joint_ratio and generate_joint_score:
                    x, r_xz, t_xz = simulator.rvs_ratio_score(
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
                    x, t_xz = simulator.rvs_score(
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
                    all_r_xz += list(r_xz)
                if generate_joint_score:
                    all_t_xz += list(t_xz)
                if checkpoint and (generate_joint_ratio or generate_joint_score):
                    all_z_checkpoints += list(z_checkpoints)
                    if generate_joint_ratio:
                        all_r_xz_checkpoints += list(r_xz_checkpoints)
                    if generate_joint_score:
                        all_t_xz_checkpoints += list(t_xz_checkpoints)

            except SimulatorException as e:
                logging.warning('Simulator raised exception: %s', e)

                if continue_after_exceptions:
                    logging.info('Ignoring this parameter point and continuing with others.')
                else:
                    raise

    all_theta0 = np.array(all_theta0)
    all_theta1 = np.array(all_theta1)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    if generate_joint_ratio:
        all_r_xz = np.array(all_r_xz)
    if generate_joint_score:
        all_t_xz = np.array(all_t_xz)
    if checkpoint and (generate_joint_ratio or generate_joint_score):
        all_z_checkpoints = np.array(all_z_checkpoints)
        if generate_joint_ratio:
            all_r_xz_checkpoints = np.array(all_r_xz_checkpoints)
        if generate_joint_score:
            all_t_xz_checkpoints = np.array(all_t_xz_checkpoints)

    # Debug output
    for i_event in range(min(10, len(all_z_checkpoints))):
        logging.debug('Checkpoint information for event %s:', i_event + 1)
        for i_checkpoint in range(all_z_checkpoints.shape[1]):
            logging.debug('  CP %s: z = %s, r = %s, t = %s',
                          i_checkpoint + 1,
                          all_z_checkpoints[i_event, i_checkpoint],
                          all_r_xz_checkpoints[i_event, i_checkpoint],
                          all_t_xz_checkpoints[i_event, i_checkpoint])
        logging.debug('Sum:   r = %s, t = %s', np.prod(all_r_xz_checkpoints[i_event]),
                      np.sum(all_t_xz_checkpoints[i_event], axis=0))
        logging.debug('Total: r = %s, t = %s', all_r_xz[i_event], all_t_xz[i_event])

    logging.info('Saving results')

    # Save results
    np.save(folder + '/theta0_' + filename + '.npy', all_theta0)
    np.save(folder + '/theta1_' + filename + '.npy', all_theta1)
    np.save(folder + '/x_' + filename + '.npy', all_x)
    np.save(folder + '/y_' + filename + '.npy', all_y)
    if generate_joint_ratio:
        np.save(folder + '/r_xz_' + filename + '.npy', all_r_xz)
    if generate_joint_score:
        np.save(folder + '/t_xz_' + filename + '.npy', all_t_xz)

    if checkpoint and (generate_joint_ratio or generate_joint_score):
        np.save(folder + '/z_checkpoints_' + filename + '.npy', all_z_checkpoints)
    if generate_joint_ratio:
        np.save(folder + '/r_xz_checkpoints_' + filename + '.npy', all_r_xz_checkpoints)
    if generate_joint_score:
        np.save(folder + '/t_xz_checkpoints_' + filename + '.npy', all_t_xz_checkpoints)


def main():
    """ Starts simulation """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator',
                        help='Simulator: "gaussian", "galton", "epidemiology", "epidemiology2d", "lotkavolterra", '
                        'or "randomwalk".')
    parser.add_argument('sample', help='Sample label (like "train" or "test")')
    parser.add_argument('--checkpoint', action='store_true', help='Checkpoint z states')
    parser.add_argument('--theta0', default=None, help='Theta0 file, defaults to standard parameters')
    parser.add_argument('--theta1', default=None, help='Theta1 file, defaults to standard parameters')
    parser.add_argument('--singletheta', action='store_true', help='If argument theta0 is not set, generates sample'
                                                                   + ' for one reference theta rather than a set')
    parser.add_argument('--gridsampling', action='store_true', help='If argument theta0 is not set, samples theta0 on a'
                                                                    + ' grid rather than randomly')
    parser.add_argument('--nthetas', type=int, default=1000, help='If argument theta0 is not set and singletheta is'
                                                                  ' False (the default), sets the number of theta'
                                                                  'benchmarks samples')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of samples per theta value')
    parser.add_argument('--noratio', action='store_true', help='Do not generate joint ratio')
    parser.add_argument('--noscore', action='store_true', help='Do not generate joint score')
    parser.add_argument('--debug', action='store_true', help='Print debug output')

    args = parser.parse_args()

    # Set up logging and numpy
    general_init(debug=args.debug)

    # Start simulation
    simulate(
        args.simulator,
        args.sample,
        checkpoint=args.checkpoint,
        theta0=args.theta0,
        theta1=args.theta1,
        single_theta=args.singletheta,
        n_thetas=args.nthetas,
        n_samples_per_theta=args.nsamples,
        generate_joint_ratio=not args.noratio,
        generate_joint_score=not args.noscore,
        grid_sampling=args.gridsampling,
        draw_from=[0]
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
