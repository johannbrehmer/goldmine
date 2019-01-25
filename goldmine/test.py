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
          checkpoint=False,
         run=0,
         alpha=1.,
         model_label='model',
         trained_on_single_theta=False,
         training_sample_size=None,
         test_sample='test',
         evaluate_densities_on_original_theta=True,
         evaluate_densities_on_grid=False,
         evaluate_ratios_on_grid=False,
         evaluate_score_on_original_theta=False,
         theta_grid=None,
         theta1_grid=None,
         generate_samples=False,
         discretize_generated_samples=False,
         grid_n_samples=1000,
         classify_surrogate_vs_true_samples=False):
    """ Main evaluation function """

    logging.info('Starting evaluation')
    logging.info('  Simulator:                        %s', simulator_name)
    logging.info('  Inference method:                 %s', inference_name)
    logging.info('  Checkpoint:                       %s', checkpoint)
    logging.info('  ML model name:                    %s', model_label)
    logging.info('  Run number:                       %s', run)
    logging.info('  Test sample:                      %s', test_sample)
    logging.info('  alpha:                            %s', alpha)
    logging.info('  Single-theta tr. sample:          %s', trained_on_single_theta)
    logging.info('  Training sample size:             %s',
                 'maximal' if training_sample_size is None else training_sample_size)
    logging.info('  Evaluate log p on original theta: %s', evaluate_densities_on_original_theta)
    logging.info('  Evaluate log p on grid:           %s', evaluate_densities_on_grid)
    logging.info('  Evaluate ratios on grid:          %s', evaluate_ratios_on_grid)
    if evaluate_densities_on_grid or evaluate_ratios_on_grid:
        if theta_grid is None:
            logging.info('  Theta grid:                       default grid with default resolution')
        elif isinstance(theta_grid, int):
            logging.info('  Theta grid:                       default grid with %s points per dimension', theta_grid)
        else:
            logging.info('  Theta grid:                       %s', theta_grid[0])
            for grid_component in theta_grid[1:]:
                logging.info('                                    %s', grid_component)
    if evaluate_ratios_on_grid:
        if theta1_grid is None:
            logging.info('  Denominator theta:                default')
        else:
            logging.info('  Denominator theta:                %s', theta1_grid)
    logging.info('  Grid x points saved:              %s', grid_n_samples)
    logging.info('  Generate samples:                 %s', generate_samples)
    logging.info('  Discretize samples                %s', discretize_generated_samples)
    logging.info('  Classify samples vs true:         %s', classify_surrogate_vs_true_samples)

    # Check paths
    create_missing_folders(base_dir, simulator_name, inference_name)

    # Folders
    sample_folder = base_dir + '/goldmine/data/samples/' + simulator_name
    model_folder = base_dir + '/goldmine/data/models/' + simulator_name + '/' + inference_name
    result_folder = base_dir + '/goldmine/data/results/' + simulator_name + '/' + inference_name

    # Filenames
    model_filename = model_label
    result_filename = ''
    if checkpoint:
        model_filename += '_checkpoint'
    if model_label != 'model':
        result_filename = '_' + model_label
    if trained_on_single_theta:
        model_filename += '_singletheta'
        result_filename += '_trainedonsingletheta'
    if training_sample_size is not None:
        model_filename += '_trainingsamplesize_' + str(training_sample_size)
        result_filename += '_trainingsamplesize_' + str(training_sample_size)
    test_filename = test_sample

    if run is None:
        run_appendix = ''
    elif int(run) == 0:
        run_appendix = ''
    else:
        run_appendix = '_run' + str(int(run))
    model_filename += run_appendix
    result_filename += run_appendix

    # Theta grid
    simulator = None
    if (evaluate_densities_on_grid or evaluate_ratios_on_grid) and (theta_grid is None or isinstance(theta_grid, int)):
        simulator = create_simulator(simulator_name)
        theta_grid = simulator.theta_grid_default(n_points_per_dim=theta_grid)

    if evaluate_ratios_on_grid and theta1_grid is None:
        if simulator is None:
            simulator = create_simulator(simulator_name)
        _, theta1_grid = simulator.theta_defaults(single_theta=True)
        theta1_grid = theta1_grid[0]

    # # Load train data
    # logging.info('Loading many-theta  train sample')
    # thetas_train = load_and_check(sample_folder + '/theta0_train.npy')
    # xs_train = load_and_check(sample_folder + '/x_train.npy')
    #
    # n_samples_train = xs_train.shape[0]
    # n_observables_train = xs_train.shape[1]
    # n_parameters_train = thetas_train.shape[1]
    # assert thetas_train.shape[0] == n_samples_train
    #
    # logging.info('Found %s samples with %s parameters and %s observables',
    #              n_samples_train, n_parameters_train, n_observables_train)

    # Load test data
    logging.info('Loading many-theta test sample')
    thetas_test = load_and_check(sample_folder + '/theta0_' + test_filename + '.npy')
    xs_test = load_and_check(sample_folder + '/x_' + test_filename + '.npy')

    n_samples = xs_test.shape[0]
    n_observables = xs_test.shape[1]
    n_parameters = thetas_test.shape[1]
    assert thetas_test.shape[0] == n_samples

    logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

    # Load test data (single theta)
    logging.info('Loading single-theta test sample')
    thetas_singletheta = load_and_check(sample_folder + '/theta0_' + test_filename + '_singletheta.npy')
    xs_singletheta = load_and_check(sample_folder + '/x_' + test_filename + '_singletheta.npy')

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
            # logging.info('Estimating densities on train sample')
            # log_p_hat = inference.predict_density(thetas_train, xs_train, log=True)
            # np.save(
            #     result_folder + '/log_p_hat_train' + result_filename + '.npy',
            #     log_p_hat
            # )

            logging.info('Estimating densities on many-theta test sample')
            log_p_hat = inference.predict_density(thetas_test, xs_test, log=True)
            np.save(
                result_folder + '/log_p_hat_' + test_filename + result_filename + '.npy',
                log_p_hat
            )

            logging.info('Estimating densities on single-theta test sample, testing original theta')
            log_p_hat = inference.predict_density(thetas_singletheta, xs_singletheta, log=True)
            np.save(
                result_folder + '/log_p_hat_' + test_filename + '_singletheta' + result_filename + '.npy',
                log_p_hat
            )

        except NotImplementedError:
            logging.warning('Inference method %s does not support density evaluation', inference_name)

    if evaluate_densities_on_grid:
        try:
            logging.info('Estimating densities on single-theta test sample, testing theta grid')

            theta_grid_points = np.meshgrid(*theta_grid, indexing='ij')
            theta_grid_points = np.array(theta_grid_points).reshape((len(theta_grid), -1))
            theta_grid_points = theta_grid_points.T

            log_p_hat_grid = []

            for theta in theta_grid_points:
                logging.debug('Grid point %s', theta)
                log_p_hat_grid.append(inference.predict_density(theta, xs_singletheta[:grid_n_samples], log=True))

            np.save(
                result_folder + '/theta_grid.npy',
                theta_grid_points
            )
            log_p_hat_grid = np.asarray(log_p_hat_grid)
            np.save(
                result_folder + '/log_p_hat_' + test_filename + '_singletheta_evaluated_on_grid_' + result_filename
                + '.npy',
                log_p_hat_grid
            )

        except NotImplementedError:
            logging.warning('Inference method %s does not support density evaluation', inference_name)

    if evaluate_ratios_on_grid:
        try:
            logging.info('Estimating ratios on single-theta test sample, testing theta0 grid')

            theta_grid_points = np.meshgrid(*theta_grid, indexing='ij')
            theta_grid_points = np.array(theta_grid_points).reshape((len(theta_grid), -1))
            theta_grid_points = theta_grid_points.T

            log_r_hat_grid = []

            for theta in theta_grid_points:
                logging.debug('Grid point %s vs %s', theta, theta1_grid)
                log_r_hat_grid.append(inference.predict_ratio(theta, theta1_grid, xs_singletheta[:grid_n_samples],
                                                              log=True))

            np.save(
                result_folder + '/theta_grid.npy',
                theta_grid_points
            )
            log_r_hat_grid = np.asarray(log_r_hat_grid)
            np.save(
                result_folder + '/log_r_hat_' + test_filename + '_singletheta_evaluated_on_grid_' + result_filename
                + '.npy',
                log_r_hat_grid
            )

        except NotImplementedError:
            logging.warning('Inference method %s does not support ratio evaluation', inference_name)

    if evaluate_score_on_original_theta:
        try:
            # logging.info('Estimating score on train sample')
            # t_hat = inference.predict_score(thetas_train, xs_train)
            # np.save(
            #     result_folder + '/t_hat_train' + result_filename + '.npy',
            #     t_hat
            # )

            logging.info('Estimating score on many-theta test sample')
            t_hat = inference.predict_score(thetas_test, xs_test)
            np.save(
                result_folder + '/t_hat_' + test_filename + result_filename + '.npy',
                t_hat
            )

            logging.info('Estimating score on single-theta test sample, testing original theta')
            t_hat = inference.predict_score(thetas_singletheta, xs_singletheta)
            np.save(
                result_folder + '/t_hat_' + test_filename + '_singletheta' + result_filename + '.npy',
                t_hat
            )

        except NotImplementedError:
            logging.warning('Inference method %s does not support score evaluation', inference_name)

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
        roc_auc, tpr, fpr = discriminate_samples(xs_singletheta, xs_surrogate)
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

    # General setup
    parser.add_argument('simulator',
                        help='Simulator: "gaussian", "galton", "epidemiology", "epidemiology2d", "lotkavolterra"')
    parser.add_argument('inference', help='Inference method: "histogram", "maf", "scandal", "rascandal", "scandalcv"')
    parser.add_argument('--checkpoint', action='store_true', help='Checkpoint z states')
    parser.add_argument('--model', type=str, default='model',
                        help='Trained model name. Default: "model".')
    parser.add_argument('--testsample', type=str, default='test',
                        help='Label (filename) for the test sample.')
    parser.add_argument('-i', type=int, default=0,
                        help='Run number for multiple repeated trainings.')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter for SCANDAL')
    parser.add_argument('--singletheta', action='store_true', help='Use model trained on single-theta sample.')
    parser.add_argument('--samplesize', type=int, default=None,
                        help='Number of (training + validation) samples considered')

    # Metrics
    parser.add_argument('--density', action='store_true',
                        help='Evaluate density on original theta')
    parser.add_argument('--densitygrid', action='store_true',
                        help='Evaluate density on theta grid')
    parser.add_argument('--ratiogrid', action='store_true',
                        help='Evaluate likelihood ratio on theta grid')
    parser.add_argument('--score', action='store_true',
                        help='Evaluate score on original theta')
    parser.add_argument('--classifiertest', action='store_true',
                        help='Train classifier to discriminate between samples from simulator and surrogate')

    # Options
    parser.add_argument('--gridnx', default=1000, type=int,
                        help='Number of phase-space points saved for the grid evaluation. Default: 1000.')

    args = parser.parse_args()

    # Start simulation
    test(
        args.simulator,
        args.inference,
        checkpoint=args.checkpoint,
        model_label=args.model,
        test_sample=args.testsample,
        run=args.i,
        alpha=args.alpha,
        trained_on_single_theta=args.singletheta,
        training_sample_size=args.samplesize,
        evaluate_densities_on_original_theta=args.density,
        evaluate_densities_on_grid=args.densitygrid,
        evaluate_ratios_on_grid=args.ratiogrid,
        evaluate_score_on_original_theta=args.score,
        generate_samples=args.classifiertest,
        classify_surrogate_vs_true_samples=args.classifiertest,
        grid_n_samples=args.gridnx
    )

    logging.info("That's all for now, have a nice day!")


if __name__ == '__main__':
    main()
