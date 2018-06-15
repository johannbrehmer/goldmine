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
          alpha=1.,
          training_sample_size=None,
          n_epochs=50,
          batch_size=64,
          initial_lr=0.001,
          final_lr=0.0001):
    """ Main training function """

    logging.info('Starting training')
    logging.info('  Simulator:            %s', simulator_name)
    logging.info('  Inference method:     %s', inference_name)
    logging.info('  alpha:                %s', alpha)
    logging.info('  Training sample size: %s',
                 'maximal' if training_sample_size is None else training_sample_size)
    logging.info('  Epochs:               %s', n_epochs)
    logging.info('  Batch size:           %s', batch_size)
    logging.info('  Learning rate:        %s initially, decaying to %s', initial_lr, final_lr)

    # Folders and filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    sample_filename = simulator_name + '_train'
    output_filename = simulator_name + '_' + inference_name
    if training_sample_size is not None:
        output_filename += '_trainingsamplesize_' + str(training_sample_size)

    # Load training data and creating model
    logging.info('Loading %s training data from %s', simulator_name, sample_folder + '/' + sample_filename + '_*.npy')
    thetas = np.load(sample_folder + '/' + sample_filename + '_theta0.npy')
    xs = np.load(sample_folder + '/' + sample_filename + '_x.npy')

    n_samples = thetas.shape[0]
    n_parameters = thetas.shape[1]
    n_observables = xs.shape[1]

    logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

    inference = create_inference(
        inference_name,
        n_parameters=n_parameters,
        n_observables=n_observables,
        alpha=alpha
    )

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

        thetas = thetas[:training_sample_size]
        xs = xs[:training_sample_size]
        if ys is not None:
            ys = ys[:training_sample_size]
        if r_xz is not None:
            r_xz = r_xz[:training_sample_size]
        if t_xz is not None:
            t_xz = t_xz[:training_sample_size]

        logging.info('Only using %s of %s training samples', xs.shape[0], n_samples)

    # Train model
    logging.info('Training model %s on %s data', inference_name, simulator_name)
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
    logging.info('Saving learned model to %s', model_folder + '/' + output_filename + '.pt')
    inference.save(model_folder + '/' + output_filename + '.pt')


def main():
    """ Starts training """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator', help='Simulator: "galton" or "epidemiology"')
    parser.add_argument('inference', help='Inference method: "maf" or "scandal"')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter for SCANDAL')
    parser.add_argument('--trainingsamplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered')

    args = parser.parse_args()

    # Start simulation
    train(
        args.simulator,
        args.inference,
        alpha=args.alpha,
        training_sample_size=args.trainingsamplesize
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
