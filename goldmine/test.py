#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path
import numpy as np

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from goldmine.various.look_up import create_inference, create_simulator
    from goldmine.various.utils import general_init, load_and_check, discretize, create_missing_folders
    from goldmine.various.discriminate_samples import discriminate_samples
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)

    from goldmine.various.look_up import create_inference, create_simulator
    from goldmine.various.utils import general_init, load_and_check, discretize, create_missing_folders
    from goldmine.various.discriminate_samples import discriminate_samples


def test(simulator_name,
         inference_name,
         run=0,
         alpha=1.,
         model_label='model',
         trained_on_single_theta=False,
         training_sample_size=None,
         evaluate_densities_on_original_theta=True,
         evaluate_densities_on_grid=False,
         evaluate_ratios_on_grid=False,
         theta_grid=10,
         generate_samples=False,
         discretize_generated_samples=False,
         classify_surrogate_vs_true_samples=False):
    """ Main evaluation function """

    logging.info('Starting evaluation')
    logging.info('  Simulator:                        %s', simulator_name)
    logging.info('  Inference method:                 %s', inference_name)
    logging.info('  ML model name:                    %s', model_label)
    logging.info('  Run number:                       %s', run)
    logging.info('  alpha:                            %s', alpha)
    logging.info('  Single-theta tr. sample:          %s', trained_on_single_theta)
    logging.info('  Training sample size:             %s',
                 'maximal' if training_sample_size is None else training_sample_size)
    logging.info('  Evaluate log p on original theta: %s', evaluate_densities_on_original_theta)
    logging.info('  Evaluate log p on grid:           %s', evaluate_densities_on_grid)
    logging.info('  Evaluate ratios on grid:          %s', evaluate_ratios_on_grid)
    if evaluate_densities_on_grid:
        if isinstance(theta_grid, int):
            logging.info('  Theta grid:                       default grid with %s points per dimension', theta_grid)
        else:
            logging.info('  Theta grid:                       %s', theta_grid[0])
            for grid_component in theta_grid[1:]:
                logging.info('                                    %s', grid_component)
    logging.info('  Generate samples:                 %s', generate_samples)
    logging.info('  Discretize samples                %s', discretize_generated_samples)
    logging.info('  Classify samples vs true:         %s', classify_surrogate_vs_true_samples)

    # Check paths
    create_missing_folders(base_dir, simulator_name, inference_name)

    # Folders and filenames
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    model_filename = model_label
    result_filename = ''
    if trained_on_single_theta:
        model_filename += '_singletheta'
        result_filename += '_trainedonsingletheta'
    if training_sample_size is not None:
        model_filename += '_trainingsamplesize_' + str(training_sample_size)
        result_filename += '_trainingsamplesize_' + str(training_sample_size)

    if run is None:
        run_appendix = ''
    elif int(run) == 0:
        run_appendix = ''
    else:
        run_appendix = '_run' + str(int(run))
    model_filename += run_appendix
    result_filename += run_appendix

    # Theta grid
    if evaluate_densities_on_grid or evaluate_ratios_on_grid:
        if isinstance(theta_grid, int):
            simulator = create_simulator(simulator_name)
            theta_grid = simulator.theta_grid_default(n_points_per_dim=theta_grid)

    # Load train data
    logging.info('Loading many-theta  train sample')
    thetas_train = load_and_check(sample_folder + '/theta0_train.npy')
    xs_train = load_and_check(sample_folder + '/x_train.npy')

    n_samples_train = xs_train.shape[0]
    n_observables_train = xs_train.shape[1]
    n_parameters_train = thetas_train.shape[1]
    assert thetas_train.shape[0] == n_samples_train

    logging.info('Found %s samples with %s parameters and %s observables',
                 n_samples_train, n_parameters_train, n_observables_train)

    # Load test data
    logging.info('Loading many-theta test sample')
    thetas_test = load_and_check(sample_folder + '/theta0_test.npy')
    xs_test = load_and_check(sample_folder + '/x_test.npy')

    n_samples = xs_test.shape[0]
    n_observables = xs_test.shape[1]
    n_parameters = thetas_test.shape[1]
    assert thetas_test.shape[0] == n_samples

    logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

    # Load test data (single theta)
    logging.info('Loading single-theta test sample')
    thetas_singletheta = load_and_check(sample_folder + '/theta0_test_singletheta.npy')
    xs_singletheta = load_and_check(sample_folder + '/x_test_singletheta.npy')

    n_samples_singletheta = xs_singletheta.shape[0]
    n_observables_singletheta = xs_singletheta.shape[1]
    n_parameters_singletheta = thetas_singletheta.shape[1]
    assert thetas_singletheta.shape[0] == n_samples_singletheta

    logging.info('Found %s samples with %s parameters and %s observables',
                 n_samples_singletheta, n_parameters_singletheta, n_observables_singletheta)

    # Load inference model
    logging.info('Loading trained model from %s', model_folder + '/' + model_filename + '.*')
    inference = create_inference(
        inference_name,
        filename=model_folder + '/' + model_filename
    )

    # Evaluate density on test sample
    if evaluate_densities_on_original_theta:
        try:
            logging.info('Estimating densities on train sample')
            log_p_hat = inference.predict_density(thetas_train, xs_train, log=True)
            np.save(
                result_folder + '/log_p_hat_train' + result_filename + '.npy',
                log_p_hat
            )

            logging.info('Estimating densities on many-theta test sample')
            log_p_hat = inference.predict_density(thetas_test, xs_test, log=True)
            np.save(
                result_folder + '/log_p_hat_test' + result_filename + '.npy',
                log_p_hat
            )

            logging.info('Estimating densities on single-theta test sample, testing original theta')
            log_p_hat = inference.predict_density(thetas_singletheta, xs_singletheta, log=True)
            np.save(
                result_folder + '/log_p_hat_test_singletheta' + result_filename + '.npy',
                log_p_hat
            )

        except NotImplementedError:
            logging.warning('Inference method %s does not support density evaluation', inference_name)

    # TODO
    if evaluate_densities_on_grid:
        try:
            logging.info('Estimating densities on single-theta test sample, testing theta grid')

            raise NotImplementedError

            # Loop over theta grid
            #   For each theta, call:
            #   log_p_hat = inference.predict_density(thetas_singletheta, xs_singletheta, log=True)
            # Save results

        except NotImplementedError:
            logging.warning('Inference method %s does not support density evaluation', inference_name)

    # TODO
    if evaluate_ratios_on_grid:
        raise NotImplementedError('Likelihood ratio evaluation on grid not implemented yet')

    # Generate samples
    if generate_samples:
        logging.info('Generating samples according to learned density')
        try:
            xs_surrogate = inference.generate_samples(thetas_singletheta)

            if discretize_generated_samples:
                discretization = create_simulator(simulator_name).get_discretization()

                logging.info('Discretizing data with scheme %s', discretization)
                xs_surrogate = discretize(xs_surrogate, discretization)

            np.save(
                result_folder + '/samples_from_p_hat' + result_filename + '.npy',
                xs_surrogate
            )
        except NotImplementedError:
            logging.warning('Inference method %s does not support sample generation', inference_name)

    # Train classifier to distinguish samples from surrogate from samples from simulator
    if classify_surrogate_vs_true_samples:
        logging.info('Training classifier to discriminate surrogate samples from simulator samples')
        xs_surrogate = load_and_check(
            result_folder + '/samples_from_p_hat' + result_filename + '.npy'
        )
        roc_auc, tpr, fpr = discriminate_samples(xs_test, xs_surrogate)
        np.save(
            result_folder + '/roc_auc_surrogate_vs_simulator' + result_filename + '.npy',
            [roc_auc]
        )
        np.save(
            result_folder + '/fpr_surrogate_vs_simulator' + result_filename + '.npy',
            [fpr]
        )
        np.save(
            result_folder + '/tpr_surrogate_vs_simulator' + result_filename + '.npy',
            [tpr]
        )


def main():
    """ Starts training """

    # Set up logging and numpy
    general_init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments with gold from the simulator')

    parser.add_argument('simulator',
                        help='Simulator: "gaussian", "galton", "epidemiology", "epidemiology2d", "lotkavolterra"')
    parser.add_argument('inference', help='Inference method: "histogram", "maf", or "scandal"')
    parser.add_argument('-i', type=int, default=0,
                        help='Run number for multiple repeated trainings.')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter for SCANDAL')
    parser.add_argument('--singletheta', action='store_true', help='Use model trained on single-theta sample.')
    parser.add_argument('--samplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered')
    parser.add_argument('--classifiertest', action='store_true',
                        help='Train classifier to discriminate between samples from simulator and surrogate')

    args = parser.parse_args()

    # Start simulation
    test(
        args.simulator,
        args.inference,
        run=args.i,
        alpha=args.alpha,
        trained_on_single_theta=args.singletheta,
        training_sample_size=args.samplesize,
        evaluate_densities_on_original_theta=True,
        generate_samples=args.classifiertest,
        classify_surrogate_vs_true_samples=args.classifiertest,
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
