#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_simulator
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    print(sys.path)
    from goldmine.various.look_up import create_simulator


def test(simulator_name,
         inference_name,
         sample_folder,
         sample_filename,
         model_folder,
         model_filename,
         n_epochs=50,
         batch_size=32,
         initial_lr=0.001,
         final_lr=0.0001,
         random_state=None):

    """ Main training function """

    pass



def run_test():
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

    logging.info('Starting simulation')
    logging.info('  Simulator:            %s', args.simulator)
    logging.info('  Inference method:     %s', args.inference)

    # Validate inputs

    # Filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + args.simulator
    sample_filename = args.simulator + '_' + args.sample
    model_folder = base_dir + '/goldmine/data/models/' + args.simulator + '/' + args.inference
    model_filename = args.simulator + '_' + args.inference

    # Start simulation
    test(
        args.simulator,
        args.inference,
        sample_folder,
        sample_filename,
        model_folder,
        model_filename
    )


if __name__ == '__main__':
    run_test()
