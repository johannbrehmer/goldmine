#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle, load_and_check, create_missing_folders
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init, shuffle, load_and_check, create_missing_folders


def train(simulator_name,
          inference_name,
          checkpoint=False,
          model_label='model',
          run=0,
          n_components=1,
          n_mades=3,
          hidden_layers=2,
          units_per_layer=20,
          batch_norm=False,
          activation='relu',
          n_bins_theta='auto',
          histogram_observables='all',
          n_bins_x='auto',
          separate_1d_x_histos=False,
          fill_empty_bins=False,
          alpha=0.0001,
          beta=0.0001,
          training_sample='train',
          single_theta=False,
          training_sample_size=None,
          n_epochs=20,
          compensate_sample_size=False,
          batch_size=128,
          trainer='adam',
          initial_lr=0.001,
          final_lr=0.0001,
          validation_split=0.2,
          early_stopping=True):
    """ Main training function """

    if single_theta:
        n_bins_theta = 1
    if histogram_observables is None:
        histogram_observables = 'all'
    if len(histogram_observables) == 0:
        histogram_observables = 'all'

    logging.info('Starting training')
    logging.info('  Simulator:             %s', simulator_name)
    logging.info('  Inference method:      %s', inference_name)
    logging.info('  Checkpoint:            %s', checkpoint)
    logging.info('  ML model name:         %s', model_label)
    logging.info('  Run number:            %s', run)
    logging.info('  Mixture components:    %s', n_components)
    logging.info('  MADEs:                 %s', n_mades)
    logging.info('  Hidden layers:         %s', hidden_layers)
    logging.info('  Units / layer:         %s', units_per_layer)
    logging.info('  Batch norm:            %s', batch_norm)
    logging.info('  Activation function:   %s', activation)
    logging.info('  Histogram theta bins:  %s', n_bins_theta)
    logging.info('  Histogram observables: %s', histogram_observables)
    logging.info('  Histogram x bins:      %s', separate_1d_x_histos)
    logging.info('  1d x histograms:       %s', n_bins_x)
    logging.info('  Fill empty bins:       %s', fill_empty_bins)
    logging.info('  SCANDAL/RASCAL alpha:  %s', alpha)
    logging.info('  RASCANDAL beta:        %s', beta)
    logging.info('  Training sample name:  %s', training_sample)
    logging.info('  Train on single theta: %s', single_theta)
    logging.info('  Training sample size:  %s',
                 'maximal' if training_sample_size is None else training_sample_size)
    if compensate_sample_size and training_sample_size is not None:
        logging.info('  Epochs:                %s (plus compensation for decreased sample size)', n_epochs)
    else:
        logging.info('  Epochs:                %s', n_epochs)
    logging.info('  Batch size:            %s', batch_size)
    logging.info('  Optimizer:             %s', trainer)
    logging.info('  Learning rate:         %s initially, decaying to %s', initial_lr, final_lr)
    logging.info('  Validation split:      %s', validation_split)
    logging.info('  Early stopping:        %s', early_stopping)

    # Check paths
    create_missing_folders(base_dir, simulator_name, inference_name)

    # Folders and filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    sample_filename = training_sample
    output_filename = model_label
    if checkpoint:
        output_filename += '_checkpoint'
    if single_theta:
        output_filename += '_singletheta'
        sample_filename += '_singletheta'
    if training_sample_size is not None:
        output_filename += '_trainingsamplesize_' + str(training_sample_size)

    if run is None:
        run_appendix = ''
    elif int(run) == 0:
        run_appendix = ''
    else:
        run_appendix = '_run' + str(int(run))
    output_filename += run_appendix

    # Load training data and creating model
    logging.info('Loading %s training data from %s', simulator_name, sample_folder + '/*_' + sample_filename + '.npy')
    thetas = load_and_check(sample_folder + '/theta0_' + sample_filename + '.npy')
    xs = load_and_check(sample_folder + '/x_' + sample_filename + '.npy')
    z_checkpoints = None
    if checkpoint:
        z_checkpoints = load_and_check(sample_folder + '/z_checkpoints_' + sample_filename + '.npy')

    n_samples = thetas.shape[0]
    n_parameters = thetas.shape[1]
    n_observables = xs.shape[1]
    n_latent = None
    if checkpoint:
        n_latent = z_checkpoints.shape[2]

    if checkpoint:
        logging.info('Found %s samples with %s parameters, %s observables, and checkpoints with %s latent variables',
                     n_samples, n_parameters, n_observables, n_latent)
    else:
        logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

    inference = create_inference(
        inference_name,
        checkpoint=checkpoint,
        n_mades=n_mades,
        n_components=n_components,
        n_made_hidden_layers=hidden_layers,
        n_made_units_per_layer=units_per_layer,
        n_hidden_layers=hidden_layers,
        n_units_per_layer=units_per_layer,
        batch_norm=batch_norm,
        activation=activation,
        n_parameters=n_parameters,
        n_observables=n_observables,
        n_latent=n_latent,
        n_bins_theta=n_bins_theta,
        n_bins_x=n_bins_x,
        separate_1d_x_histos=separate_1d_x_histos,
        observables=histogram_observables
    )

    # Load more data
    if inference.requires_class_label():
        ys = load_and_check(sample_folder + '/y_' + sample_filename + '.npy')
    else:
        ys = None

    if inference.requires_joint_ratio():
        r_xz = load_and_check(sample_folder + '/r_xz_' + sample_filename + '.npy')
        theta1 = load_and_check(sample_folder + '/theta1_' + sample_filename + '.npy')
        if len(theta1.shape) > 1:  # For now, we just want one constant theta1. Might be changed in later versions
            theta1 = theta1[0]
        assert theta1.shape == thetas[0].shape, 'Shape mismatch between theta0 and theta1'
    else:
        r_xz = None
        theta1 = None

    if inference.requires_joint_score():
        t_xz = load_and_check(sample_folder + '/t_xz_' + sample_filename + '.npy')
    else:
        t_xz = None

    r_xz_checkpoints = None
    t_xz_checkpoints = None
    if checkpoint:
        if inference.requires_joint_ratio():
            r_xz_checkpoints = load_and_check(sample_folder + '/r_xz_checkpoints_' + sample_filename + '.npy')
        if inference.requires_joint_score():
            t_xz_checkpoints = load_and_check(sample_folder + '/t_xz_checkpoints_' + sample_filename + '.npy')

    # Restricted training sample size
    if training_sample_size is not None and training_sample_size < n_samples:
        thetas, xs, ys, r_xz, t_xz, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints = shuffle(
            thetas, xs, ys, r_xz, t_xz, z_checkpoints, r_xz_checkpoints, t_xz_checkpoints
        )

        thetas = thetas[:training_sample_size]
        xs = xs[:training_sample_size]
        if ys is not None:
            ys = ys[:training_sample_size]
        if r_xz is not None:
            r_xz = r_xz[:training_sample_size]
        if t_xz is not None:
            t_xz = t_xz[:training_sample_size]
        if z_checkpoints is not None:
            z_checkpoints = z_checkpoints[:training_sample_size]
        if r_xz_checkpoints is not None:
            r_xz_checkpoints = r_xz_checkpoints[:training_sample_size]
        if t_xz_checkpoints is not None:
            t_xz_checkpoints = t_xz_checkpoints[:training_sample_size]

        logging.info('Only using %s of %s training samples', training_sample_size, n_samples)

        if compensate_sample_size and training_sample_size < n_samples:
            n_epochs_compensated = int(round(n_epochs * n_samples / training_sample_size, 0))
            logging.info('Compensating by increasing number of epochs from %s to %s', n_epochs, n_epochs_compensated)
            n_epochs = n_epochs_compensated

    # Train model
    logging.info('Training model %s on %s data', inference_name, simulator_name)

    if checkpoint:
        inference.fit(
            thetas, xs,
            ys, r_xz, t_xz,
            theta1=theta1,
            z_checkpoints=z_checkpoints,
            r_xz_checkpoints=r_xz_checkpoints,
            t_xz_checkpoints=t_xz_checkpoints,
            n_epochs=n_epochs,
            batch_size=batch_size,
            trainer=trainer,
            initial_learning_rate=initial_lr,
            final_learning_rate=final_lr,
            alpha=alpha,
            beta=beta,
            learning_curve_folder=result_folder,
            learning_curve_filename=output_filename,
            validation_split=validation_split,
            early_stopping=early_stopping,
            fill_empty_bins=fill_empty_bins
        )

    else:
        inference.fit(
            thetas, xs,
            ys, r_xz, t_xz,
            theta1=theta1,
            n_epochs=n_epochs,
            batch_size=batch_size,
            trainer=trainer,
            initial_learning_rate=initial_lr,
            final_learning_rate=final_lr,
            alpha=alpha,
            beta=beta,
            learning_curve_folder=result_folder,
            learning_curve_filename=output_filename,
            validation_split=validation_split,
            early_stopping=early_stopping,
            fill_empty_bins=fill_empty_bins
        )

    # Save models
    logging.info('Saving learned model to %s', model_folder + '/' + output_filename + '.*')
    inference.save(model_folder + '/' + output_filename)


def main():
    """ Starts training """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    # Basic run settings and labels
    parser.add_argument('simulator',
                        help='Simulator: "gaussian", "galton", "epidemiology", "epidemiology2d", "lotkavolterra"')
    parser.add_argument('inference', help='Inference method: "histogram", "maf", "scandal", "rascandal", "scandalcv"')
    parser.add_argument('--checkpoint', action='store_true', help='Checkpoint z states')
    parser.add_argument('--modellabel', type=str, default='model',
                        help='Additional name for the trained model.')
    parser.add_argument('--trainsample', type=str, default='train',
                        help='Label (filename) for the training sample.')
    parser.add_argument('-i', type=int, default=0,
                        help='Run number for multiple repeated trainings.')

    # Flow architecture
    parser.add_argument('--components', type=int, default=1,
                        help='Number of Gaussian mixture components in a MAF MoG. Default: 1 (a standard MAF).')
    parser.add_argument('--nades', type=int, default=5,
                        help='Number of NADEs in a MAF. Default: 5.')
    parser.add_argument('--hidden', type=int, default=1,
                        help='Number of hidden layers. Default: 1.')
    parser.add_argument('--units', type=int, default=100,
                        help='Number of units per hidden layer. Default: 100.')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Use batch normalization in the flow between MADE layers.')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function: "relu", "tanh", "sigmoid". Default is "tanh".')

    # Histogram parameters
    parser.add_argument('--thetabins', type=int, default=3,
                        help='Number of bins per parameter for histogram-based inference.')
    parser.add_argument('--observables', type=int, nargs='+',
                        help='Observable indices used for histograms.')
    parser.add_argument('--xbins', type=int, default=3,
                        help='Number of bins per observable for histogram-based inference.')
    parser.add_argument('--xhistos1d', action='store_true',
                        help='Whether to use separate 1d histograms for each observable.')
    parser.add_argument('--fillemptybins', action='store_true',
                        help='Fill empty histogram bins with 1s.')

    # Training sample details
    parser.add_argument('--singletheta', action='store_true', help='Train on single-theta sample.')
    parser.add_argument('--samplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered. Default: use all available '
                             + 'samples.')

    # Training settings
    parser.add_argument('--batchsize', type=int, default=512,
                        help='Batch size. Default: 512.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs. Default: 30.')
    parser.add_argument('--compensate_samplesize', action='store_true',
                        help='If both this option and --samplesize are used, the number of epochs is increased to'
                             + ' compensate for the decreased sample size.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='alpha parameter weighting the score MSE in the loss function of the SCANDAL, RASCAL, and'
                             'and RASCANDAL inference methods. Default: 0.0001.')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='beta parameter weighting the likelihood ratio MSE in the loss function of the RASCANDAL'
                             'inference method. Default: 0.0001.')
    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer. For now, "adam" and "sgd" are supported.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate. Default: 0.001.')
    parser.add_argument('--lrdecay', type=float, default=0.1,
                        help='Factor of learning rate decay over the whole training. Default: 0.1.')
    parser.add_argument('--validationsplit', type=float, default=0.2,
                        help='Validation split. Default: 0.2.')
    parser.add_argument('--noearlystopping', action='store_true',
                        help='Deactivate early stopping.')

    parser.add_argument('--debug', action='store_true', help='Print debug output')

    args = parser.parse_args()

    # Set up logging and numpy
    general_init(debug=args.debug)

    # Start simulation
    train(
        args.simulator,
        args.inference,
        checkpoint=args.checkpoint,
        model_label=args.modellabel,
        run=args.i,
        n_mades=args.nades,
        n_components=args.components,
        hidden_layers=args.hidden,
        units_per_layer=args.units,
        activation=args.activation,
        n_bins_theta=args.thetabins,
        n_bins_x=args.xbins,
        separate_1d_x_histos=args.xhistos1d,
        histogram_observables=args.observables,
        fill_empty_bins=args.fillemptybins,
        batch_norm=args.batchnorm,
        alpha=args.alpha,
        beta=args.beta,
        training_sample=args.trainsample,
        single_theta=args.singletheta,
        training_sample_size=args.samplesize,
        batch_size=args.batchsize,
        n_epochs=args.epochs,
        compensate_sample_size=args.compensate_samplesize,
        trainer=args.optimizer,
        initial_lr=args.lr,
        final_lr=args.lr * args.lrdecay,
        validation_split=args.validationsplit,
        early_stopping=not args.noearlystopping
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
