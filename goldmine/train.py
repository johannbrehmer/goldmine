#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    print(sys.path)
    from goldmine.various.look_up import create_inference
    from goldmine.various.utils import general_init


def train(simulator_name,
          inference_name,
          sample_folder,
          sample_filename,
          model_folder,
          model_filename,
          result_folder,
          result_filename,
          n_epochs=50,
          batch_size=32,
          initial_lr=0.001,
          final_lr=0.0001):
    """ Main training function """

    # TODO: Check inputs

    # Load training data
    thetas = np.load(sample_folder + '/' + sample_filename + '_theta0.npy')
    xs = np.load(sample_folder + '/' + sample_filename + '_x.npy')

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

    logging.debug('Array shapes: x = %s, theta = %s', xs.shape, thetas.shape)

    # Train model
    inference.fit(
        thetas, xs,
        ys, r_xz, t_xz,
        n_epochs=n_epochs,
        batch_size=batch_size,
        initial_learning_rate=initial_lr,
        final_learning_rate=final_lr
    )

    # Save models
    inference.save(model_folder + '/' + model_filename)


def run_train():
    """ Starts training """

    # Set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG,
                        datefmt='%d.%m.%Y %H:%M:%S')
    logging.info('Hi! How are you today?')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator', help='Simulator: "galton" or "epidemiology"')
    parser.add_argument('inference', help='Inference method: "maf" or "scandal"')

    args = parser.parse_args()

    logging.info('Training routine')
    logging.info('  Simulator:            %s', args.simulator)
    logging.info('  Inference method:     %s', args.inference)

    # Validate inputs

    # Filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + args.simulator
    sample_filename = args.simulator + '_train'
    model_folder = base_dir + '/goldmine/data/models/' + args.simulator + '/' + args.inference
    model_filename = args.simulator + '_' + args.inference
    result_folder = base_dir + '/goldmine/data/results/' + args.simulator + '/' + args.inference
    result_filename = args.simulator + '_' + args.inference

    # Start simulation
    train(
        args.simulator,
        args.inference,
        sample_folder,
        sample_filename,
        model_folder,
        model_filename,
        result_folder,
        result_filename
    )


if __name__ == '__main__':
    run_train()
