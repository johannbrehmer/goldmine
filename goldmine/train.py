#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    print(sys.path)
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle


def train(simulator_name,
          inference_name,
          training_sample_size=None,
          n_epochs=50,
          batch_size=64,
          initial_lr=0.001,
          final_lr=0.0001):
    """ Main training function """

    # Folders
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    # Input filenames
    sample_filename = simulator_name + '_train'

    # Output filenames
    output_filename = simulator_name + '_' + inference_name
    if training_sample_size is not None:
        output_filename += '_trainingsamplesize_' + str(training_sample_size)

    # Load training data
    thetas = np.load(sample_folder + '/' + sample_filename + '_theta0.npy')
    xs = np.load(sample_folder + '/' + sample_filename + '_x.npy')

    n_samples = thetas.shape[0]
    n_parameters = thetas.shape[1]
    n_observables = xs.shape[1]

    inference = create_inference(inference_name)(n_parameters, n_observables)

    if inference.requires_class_label():
        ys = np.load(sample_folder + '/' + sample_filename + '_y.npy')
    else:
        ys = None

    if inference.requires_joint_ratio():
        r_xz = np.load(sample_folder + '/' + sample_filename + '_r_xz.npy')
    else:
        r_xz = None

    if inference.requires_joint_score():
        t_xz = np.load(sample_folder + '/' + sample_filename + '_t_xz.npy')
    else:
        t_xz = None

    # Restricted training sample size
    if training_sample_size is not None and training_sample_size < n_samples:
        thetas, xs, ys, r_xz, t_xz = shuffle(thetas, xs, ys, r_xz, t_xz)

    logging.debug('Array shapes: x = %s, theta = %s', xs.shape, thetas.shape)

    # Train model
    inference.fit(
        thetas, xs,
        ys, r_xz, t_xz,
        n_epochs=n_epochs,
        batch_size=batch_size,
        initial_learning_rate=initial_lr,
        final_learning_rate=final_lr,
        learning_curve_folder=result_folder,
        learning_curve_filename=output_filename
    )

    # Save models
    inference.save(model_folder + '/' + output_filename)


def main():
    """ Starts training """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator', help='Simulator: "galton" or "epidemiology"')
    parser.add_argument('inference', help='Inference method: "maf" or "scandal"')
    parser.add_argument('--trainingsamplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered')

    args = parser.parse_args()

    logging.info('Start-up options:')
    logging.info('  Simulator:            %s', args.simulator)
    logging.info('  Inference method:     %s', args.inference)
    logging.info('  Training sample size: %s',
                 'maximal' if args.trainingsamplesize is None else args.trainingsamplesize)

    # Start simulation
    train(
        args.simulator,
        args.inference,
        training_sample_size=args.trainingsamplesize
    )


if __name__ == '__main__':
    main()
