#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle, load_and_check
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    print(sys.path)
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle, load_and_check


def train(simulator_name,
          inference_name,
          n_mades=3,
          n_made_hidden_layers=2,
          n_made_units_per_layer=20,
          batch_norm=False,
          activation='relu',
          n_bins='auto',
          alpha=1.,
          training_sample_size=None,
          n_epochs=20,
          compensate_sample_size=False,
          batch_size=64,
          initial_lr=0.001,
          final_lr=0.0001,
          early_stopping=True):
    """ Main training function """

    logging.info('Starting training')
    logging.info('  Simulator:            %s', simulator_name)
    logging.info('  Inference method:     %s', inference_name)
    logging.info('  MADEs:                %s', n_mades)
    logging.info('  MADE hidden layers:   %s', n_made_hidden_layers)
    logging.info('  MADE units / layer:   %s', n_made_units_per_layer)
    logging.info('  Batch norm:           %s', batch_norm)
    logging.info('  Activation function:  %s', activation)
    logging.info('  Histogram bins:       %s', n_bins)
    logging.info('  SCANDAL alpha:        %s', alpha)

    logging.info('  Training sample size: %s',
                 'maximal' if training_sample_size is None else training_sample_size)
    if compensate_sample_size and training_sample_size is not None:
        logging.info('  Epochs:               %s (plus compensation for decreased sample size)', n_epochs)
    else:
        logging.info('  Epochs:               %s', n_epochs)
    logging.info('  Batch size:           %s', batch_size)
    logging.info('  Learning rate:        %s initially, decaying to %s', initial_lr, final_lr)
    logging.info('  Early stopping:       %s', early_stopping)

    # Folders and filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    output_filename = ''
    if training_sample_size is not None:
        output_filename += '_trainingsamplesize_' + str(training_sample_size)

    # Load training data and creating model
    logging.info('Loading %s training data from %s', simulator_name, sample_folder + '/*_train.npy')
    thetas = load_and_check(sample_folder + '/theta0_train.npy')
    xs = load_and_check(sample_folder + '/x_train.npy')

    n_samples = thetas.shape[0]
    n_parameters = thetas.shape[1]
    n_observables = xs.shape[1]

    logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

    inference = create_inference(
        inference_name,
        n_mades=n_mades,
        n_made_hidden_layers=n_made_hidden_layers,
        n_made_units_per_layer=n_made_units_per_layer,
        batch_norm=batch_norm,
        activation=activation,
        n_parameters=n_parameters,
        n_observables=n_observables,
        n_bins_theta=n_bins,
        n_bins_x=n_bins
    )

    if inference.requires_class_label():
        ys = load_and_check(sample_folder + '/y_train.npy')
    else:
        ys = None

    if inference.requires_joint_ratio():
        r_xz = load_and_check(sample_folder + '/r_xz_train.npy')
    else:
        r_xz = None

    if inference.requires_joint_score():
        t_xz = load_and_check(sample_folder + '/t_xz_train.npy')
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

        logging.info('Only using %s of %s training samples', training_sample_size, n_samples)

        if compensate_sample_size and training_sample_size < n_samples:
            n_epochs_compensated = int(round(n_epochs * n_samples / training_sample_size, 0))
            logging.info('Compensating by increasing number of epochs from %s to %s', n_epochs, n_epochs_compensated)
            n_epochs = n_epochs_compensated

    # Train model
    logging.info('Training model %s on %s data', inference_name, simulator_name)
    inference.fit(
        thetas, xs,
        ys, r_xz, t_xz,
        n_epochs=n_epochs,
        batch_size=batch_size,
        initial_learning_rate=initial_lr,
        final_learning_rate=final_lr,
        alpha=alpha,
        learning_curve_folder=result_folder,
        learning_curve_filename=output_filename,
        early_stopping=early_stopping
    )

    # Save models
    logging.info('Saving learned model to %s', model_folder + '/model' + output_filename + '.*')
    inference.save(model_folder + '/model' + output_filename)


def main():
    """ Starts training """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator', help='Simulator: "gaussian", "galton", or "epidemiology"')
    parser.add_argument('inference', help='Inference method: "histogram", "maf", or "scandal"')
    parser.add_argument('--nades', type=int, default=5,
                        help='Number of NADEs in a MAF. Default: 5.')
    parser.add_argument('--hidden', type=int, default=1,
                        help='Number of hidden layers. Default: 1.')
    parser.add_argument('--units', type=int, default=100,
                        help='Number of units per hidden layer. Default: 100.')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Use batch normalization.')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function: "rely", "tanh", "sigmoid"')
    parser.add_argument('--bins', default='auto',
                        help='Number of bins per parameter and observable for histogram-based inference.')
    parser.add_argument('--samplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered. Default: use all available '
                             + 'samples.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs. Default: 20.')
    parser.add_argument('--compensate_samplesize', action='store_true',
                        help='If both this option and --samplesize are used, the number of epochs is increased to'
                             + ' compensate for the decreased sample size.')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='alpha parameter for SCANDAL. Default: 0.001.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate. Default: 0.001.')
    parser.add_argument('--lrdecay', type=float, default=0.1,
                        help='Factor of learning rate decay over the whole training. Default: 0.1.')
    parser.add_argument('--noearlystopping', action='store_true',
                        help='Deactivate early stopping.')
    parser.add_argument('--gradientclip', default=1.,
                        help='Gradient norm clipping threshold.')

    # TODO: Add option for multiple runs
    # TODO: Add option for custom filename parts
    # TODO: Better treatment of non-existent files and folders

    args = parser.parse_args()

    # Start simulation
    train(
        args.simulator,
        args.inference,
        n_mades=args.nades,
        n_made_hidden_layers=args.hidden,
        n_made_units_per_layer=args.units,
        activation=args.activation,
        n_bins=args.bins,
        batch_norm=args.batchnorm,
        alpha=args.alpha,
        training_sample_size=args.samplesize,
        n_epochs=args.epochs,
        compensate_sample_size=args.compensate_samplesize,
        initial_lr=args.lr,
        final_lr=args.lr * args.lrdecay,
        early_stopping=not args.noearlystopping
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
