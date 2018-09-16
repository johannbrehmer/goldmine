#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
from os import sys, path
import numpy as np
import re

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


def shuffle_and_combine(simulator, input_samples, output_sample, regex=False):
    logging.info('Starting shuffling and combining')
    logging.info('  Simulator:           %s', simulator)
    logging.info('  Input samples:       %s', input_samples[0])
    for sample in input_samples[1:]:
        logging.info('                       %s', sample)
    logging.info('  Output sample:       %s', output_sample)
    logging.info('  Regular expressions: %s', regex)

    # Path and filenames
    folder = base_dir + '/goldmine/data/samples/' + simulator
    filenames = ['theta0', 'theta1', 'x', 'y', 'r_xz', 't_xz']

    # Parse regular expressions
    if regex:
        input_expressions = input_samples
        input_samples = []
        for expr in input_expressions:

            logging.debug('Parsing regex %s', expr)

            regex = re.compile('x_(' + expr + ')\.npy')

            for root, _, files in os.walk(folder):
                for file in files:
                    if regex.match(file):
                        input_sample = file[2:-4]

                        if input_sample in input_samples:
                            logging.debug('  Input sample %s already in list', input_sample)
                            continue

                        logging.debug('  Found input sample %s', input_sample)
                        input_samples.append(input_sample)

    # Combine samples
    n_samples = None
    permutation = None

    for filename in filenames:

        # Load individual files
        try:
            individuals = [np.load(folder + '/' + filename + '_' + input_sample + '.npy') for input_sample in
                           input_samples]
        except FileNotFoundError:
            logging.info('Object %s does not exist for (some of the) input samples', filename)
            continue

        # Combine
        try:
            combined = np.concatenate(individuals, axis=0)
        except ValueError:
            logging.warning('Object %s: individual results do not have matching shapes!', filename)
            for input_sample, individual in zip(input_samples, individuals):
                logging.warning('  %s: %s has shape %s', input_sample, filename, individual.shape)
            continue
        logging.info('Combined %s %s files, combined shape: %s', len(individuals), filename,
                     combined.shape)

        # Shuffle
        if n_samples is None or permutation is None:
            n_samples = combined.shape[0]
            permutation = np.random.permutation(n_samples)
        else:
            if n_samples != combined.shape[0]:
                logging.error('Inconsistent shapes!')
                raise RuntimeError('Inconsistent shapes!')

        combined = combined[permutation]
        logging.info('Shuffled combined %s results', filename)

        # Save
        try:
            np.save(folder + '/' + filename + '_' + output_sample + '.npy', combined)
        except FileExistsError:
            logging.warning('File %s already exists, cannot save results!',
                            folder + '/' + filename + '_' + output_sample + '.npy')
            continue
        logging.info('Saved file %s', folder + '/' + filename + '_' + output_sample + '.npy')


def main():
    """ Shuffles different samples """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator',
                        help='Simulator: "gaussian", "galton", "epidemiology", "epidemiology2d", "lotkavolterra"')
    parser.add_argument('output', help='Combined sample label (like "train" or "test")')
    parser.add_argument('inputs', nargs='+', help='Individual input sample labels (like "train0 train1 train2"). If '
                                                  'option --regex is set, inputs can be regular expressions.')
    parser.add_argument('--regex', action='store_true', help='Allows regular expressions in inputs')

    args = parser.parse_args()

    # Start simulation
    shuffle_and_combine(
        args.simulator,
        args.output,
        args.inputs,
        args.regex
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
