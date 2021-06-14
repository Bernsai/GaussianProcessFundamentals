import os
import os.path
import gpbasics.global_parameters as global_param
global_param.ensure_init()

import matplotlib.pyplot as plt

import gpbasics.Statistics.GaussianProcess as gp

import logging
import numpy as np
import tensorflow as tf
import gpbasics.KernelBasics.Operators as op
import gpbasics.MeanFunctionBasics.MeanFunction as mf
import gpbasics.KernelBasics.PartitionOperator as po


def illustrate_prior_functions(gaussian_process: gp.AbstractGaussianProcess,
                               plt_filename: str = None, path: str = None):
    f_prior = gaussian_process.get_n_prior_functions(2, gaussian_process.kernel.get_last_hyper_parameter()).numpy()

    plt.plot(gaussian_process.data_input.data_x_test, f_prior)
    # plt.axis([0, 1, 0, 1])
    plt.title('Three samples from the GP prior')

    if plt_filename is not None and path is not None:
        logging.info("Saving plot as svg-file")
        plt.savefig(path + plt_filename, format='svg')
    else:
        plt.show()


def illustrate_posterior(dimension: int, gaussian_process: gp.AbstractGaussianProcess,
                         title: str, plt_filename: str = None, path: str = None):
    gaussian_process.covariance_matrix.reset()
    multi_dim_x_test_unsorted: np.ndarray = gaussian_process.data_input.data_x_test
    x_test_unsorted: np.ndarray = multi_dim_x_test_unsorted[:, dimension]
    y_test_unsorted: np.ndarray = gaussian_process.data_input.data_y_test
    x_test: np.ndarray = np.array(sorted(x_test_unsorted))
    multi_dim_xtest_sorted: np.ndarray = \
        np.array([x_mul for _, x_mul in
                  sorted(zip(x_test_unsorted, multi_dim_x_test_unsorted), key=lambda row: row[0])])
    y_test: np.ndarray = \
        np.array([y for _, y in sorted(zip(x_test_unsorted, y_test_unsorted), key=lambda row: row[0])])
    x_train_unsorted: np.ndarray = gaussian_process.data_input.data_x_train[:, dimension]
    y_train_unsorted: np.ndarray = gaussian_process.data_input.data_y_train
    x_train: np.ndarray = np.array(sorted(x_train_unsorted))
    y_train: np.ndarray = \
        np.array([y for _, y in sorted(zip(x_train_unsorted, y_train_unsorted), key=lambda row: row[0])])
    hyper_parameter = gaussian_process.covariance_matrix.kernel.get_last_hyper_parameter()
    noise = gaussian_process.covariance_matrix.kernel.get_noise()
    mu_unsorted: tf.Tensor = gaussian_process.aux.get_posterior_mu(hyper_parameter, noise)
    mu_: np.ndarray = \
        np.array([y for _, y in sorted(zip(x_test_unsorted, mu_unsorted), key=lambda row: row[0])])
    sd_unsorted = tf.sqrt(
        tf.linalg.diag_part(
            gaussian_process.aux.get_posterior_var(hyper_parameter, noise)))
    sd_ = np.array([y for _, y in sorted(zip(x_test_unsorted, sd_unsorted), key=lambda row: row[0])])
    mean_function: mf.MeanFunction = gaussian_process.mean_function
    plt.figure(figsize=[20, 10], dpi=600)
    plt.plot(x_train, y_train, 'b-', ms=8)
    mean_function_values: tf.Tensor = tf.reshape(
        mean_function.get_tf_tensor(mean_function.get_last_hyper_parameter(),
                                    multi_dim_xtest_sorted), [-1, ])
    values = mu_ + mean_function_values
    plt.plot(x_test, values, 'r--', lw=2)
    plt.gca().fill_between(x_test.flat, mu_ - 2 * sd_, mu_ + 2 * sd_, color="#dddddd")
    if isinstance(gaussian_process.covariance_matrix.kernel, op.ChangePointOperator) and \
            (len(gaussian_process.covariance_matrix.kernel.change_point_positions) > 0):
        plt.vlines([x[0] for x in gaussian_process.covariance_matrix.kernel.change_point_positions],
                   ymin=np.min(np.concatenate([mu_, y_test], axis=None)),
                   ymax=np.max(np.concatenate([mu_, y_test], axis=None)))

    if isinstance(gaussian_process.covariance_matrix.kernel, po.PartitionOperator) and \
            (len(gaussian_process.covariance_matrix.kernel.partitioning_model.partitioning) > 1):
        pm = gaussian_process.covariance_matrix.kernel.partitioning_model
        if gaussian_process.data_input.get_input_dimensionality() == 1:  # isinstance(pm, cpm.ChangePointModel):
            plt.vlines([criterion.cp_range[0] for criterion in pm.partitioning[1:]],
                       ymin=np.min(np.concatenate([mu_, y_test], axis=None)),
                       ymax=np.max(np.concatenate([mu_, y_test], axis=None)))

    plt.title(title)

    if plt_filename is not None and path is not None:
        logging.info("Saving plot as svg-file")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + plt_filename, format='svg')
    else:
        plt.show()
