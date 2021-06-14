import gpbasics.global_parameters as global_param

global_param.ensure_init()
import logging
from typing import List, Tuple

import tensorflow as tf

import gpbasics.KernelBasics.PartitionOperator as po
import gpbasics.Statistics.Auxiliary as ax
import gpbasics.Statistics.CovarianceMatrix as cm
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.Kernel as k
import gpbasics.KernelBasics.Operators as op
import gpbasics.MeanFunctionBasics.MeanFunction as mf

import numpy as np


class AbstractGaussianProcess:
    def __init__(self, kernel: k.Kernel, mean_function: mf.MeanFunction):
        self.mean_function: mf.MeanFunction = mean_function
        self.kernel: k.Kernel = kernel

        self.covariance_matrix: cm.CovarianceMatrix = None
        self.aux: ax.AuxiliaryGpProperties = None

        self.data_input: di.DataInput = None

        self.inducing_points: tf.Tensor = None

    def set_inducing_points(self, inducing_points: tf.Tensor):
        self.inducing_points = inducing_points

    def set_data_input(self, data_input: di.AbstractDataInput):
        self.data_input = data_input

        # Current covariance matrix object becomes obsolete with a new data input
        self.covariance_matrix.set_data_input(data_input)
        self.aux.set_data_input(data_input)

    def predict(self, kernel_hyper_param: List[tf.Tensor] = None, mean_function_hyper_param: List[tf.Tensor] = None,
                noise: tf.Tensor = None) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self.aux.reset()
        self.covariance_matrix.reset()

        if noise is None:
            noise = global_param.p_cov_matrix_jitter

        logging.debug("GP Predict: Retrieving covariance and mean function hyper parameters.")
        if mean_function_hyper_param is None:
            mean_function_hyper_param = self.mean_function.get_last_hyper_parameter()
            if mean_function_hyper_param is None:
                mean_function_hyper_param = self.mean_function.get_default_hyper_parameter()

        if kernel_hyper_param is None:
            kernel_hyper_param = self.kernel.get_last_hyper_parameter()
            if kernel_hyper_param is None:
                kernel_hyper_param = self.kernel.get_default_hyper_parameter()

        logging.debug("GP Predict: Calculating mean function values.")
        mean_function_mu = self.mean_function.get_tf_tensor(mean_function_hyper_param, self.data_input.data_x_test)

        logging.debug("GP Predict: Calculating Posterior Mu values.")
        if isinstance(self, PartitionedGaussianProcess) or isinstance(self, BlockwiseGaussianProcess):
            posterior_mus: List[tf.Tensor] = []
            index = 0
            if isinstance(self, BlockwiseGaussianProcess):
                # For BlockwiseGaussianProcesses the overarching ChangepointOperator's changepoints
                #   are treated as hyperparameters
                index = len(self.covariance_matrix.change_point_positions)
            for sub_gp in self.constituent_gps:
                num_hyp_param = sub_gp.kernel.get_number_of_hyper_parameter()
                sliced_hyper_parameter = \
                    kernel_hyper_param[index: index + num_hyp_param]
                posterior_mus.append(sub_gp.aux.get_posterior_mu(sliced_hyper_parameter, noise))
                index += num_hyp_param
            posterior_mu = tf.concat(posterior_mus, axis=0)
        else:
            posterior_mu: tf.Tensor = self.aux.get_posterior_mu(kernel_hyper_param, noise)

        full_mu = mean_function_mu + posterior_mu

        return full_mu, mean_function_mu, posterior_mu

    def get_n_prior_functions(self, n: int, hyper_param, noise):
        L_K_ss = self.covariance_matrix.get_L_K_ss(hyper_param, noise)

        mean = np.mean(self.data_input.data_y_train)

        std = np.std(self.data_input.data_y_train)

        return tf.tensordot(L_K_ss, tf.random.normal(shape=(self.data_input.n_test, n),
                                                     dtype=global_param.p_dtype, mean=mean, stddev=std), axes=1)

    def get_n_posterior_functions(self, n: int, hyper_param, noise):
        sigma_s = self.aux.get_posterior_var(hyper_param, noise)

        cholesky_sigma_s = tf.linalg.cholesky(
            tf.math.add(sigma_s, global_param.p_cov_matrix_jitter *
                        tf.eye(int(sigma_s.shape[0]), dtype=global_param.p_dtype)))

        # µ* = posterior_mu
        # f* ~ N(f* | µ~, sigma*) = µ* + cholesky_sigma_s * N(0,1)
        f_post = tf.reshape(self.aux.get_posterior_mu(hyper_param, noise), [-1, 1]) \
                 + tf.tensordot(cholesky_sigma_s, tf.random.normal(shape=(self.data_input.n_test, n),
                                                                   dtype=global_param.p_dtype), axes=1)

        return f_post

    def copy(self):
        pass


class GaussianProcess(AbstractGaussianProcess):
    def __init__(self, kernel: k.Kernel, mean_function: mf.MeanFunction):
        super(GaussianProcess, self).__init__(kernel, mean_function)
        self.covariance_matrix = cm.HolisticCovarianceMatrix(self.kernel)
        self.aux = ax.HolisticAuxiliaryGpProperties(self.covariance_matrix, self.mean_function)

    def copy(self):
        gp = GaussianProcess(self.kernel, self.mean_function)
        gp.set_inducing_points(self.inducing_points)
        return gp


class PredefinedGaussianProcess(AbstractGaussianProcess):
    def __init__(self, covariance_matrix: cm.CovarianceMatrix, mean_function: mf.MeanFunction):
        super(PredefinedGaussianProcess, self).__init__(covariance_matrix.kernel, mean_function)
        self.covariance_matrix = covariance_matrix
        self.aux = ax.HolisticAuxiliaryGpProperties(self.covariance_matrix, self.mean_function)

    def copy(self):
        gp = PredefinedGaussianProcess(self.covariance_matrix, self.mean_function)
        gp.set_inducing_points(self.inducing_points)
        return gp


class BlockwiseGaussianProcess(AbstractGaussianProcess):
    def __init__(self, kernel: op.ChangePointOperator, mean_function: mf.MeanFunction):
        super(BlockwiseGaussianProcess, self).__init__(kernel, mean_function)

        self.constituent_gps: List[GaussianProcess] = []

        for i in range(0, len(kernel.child_nodes)):
            _gp = GaussianProcess(kernel.child_nodes[i], self.mean_function)
            self.constituent_gps.append(_gp)

        self.covariance_matrix = cm.SegmentedCovarianceMatrix(kernel)

        self.aux = ax.BlockwiseAuxiliaryGpProperties(self.covariance_matrix, self.mean_function)

    def set_data_input(self, data_input: di.BlockwiseDataInput):
        assert len(data_input.data_inputs) == len(self.constituent_gps), \
            "Data Input does not fit constituent GPs of blockwise GP"
        self.data_input = data_input

        # Current covariance matrix object becomes obsolete with a new data input
        self.covariance_matrix.set_data_input(data_input)
        self.aux.set_data_input(data_input)

        for i in range(0, len(data_input.data_inputs)):
            self.constituent_gps[i].set_data_input(data_input.data_inputs[i])

    def copy(self):
        gp = BlockwiseGaussianProcess(self.kernel, self.mean_function)
        gp.set_inducing_points(self.inducing_points)
        return gp


class PartitionedGaussianProcess(AbstractGaussianProcess):
    def __init__(self, kernel: po.PartitionOperator, mean_function: mf.MeanFunction):
        super(PartitionedGaussianProcess, self).__init__(kernel, mean_function)

        self.constituent_gps: List[GaussianProcess] = []

        for i in range(0, len(kernel.child_nodes)):
            _gp = GaussianProcess(kernel.child_nodes[i], self.mean_function)
            self.constituent_gps.append(_gp)

        self.covariance_matrix = cm.SegmentedCovarianceMatrix(kernel)

        self.aux = ax.BlockwiseAuxiliaryGpProperties(self.covariance_matrix, self.mean_function)

    def set_data_input(self, data_input: di.PartitionedDataInput):
        assert len(data_input.data_inputs) == len(self.constituent_gps), \
            "Data Input does not fit constituent GPs of Partitioned GP"
        self.data_input = data_input

        # Current covariance matrix object becomes obsolete with a new data input
        self.covariance_matrix.set_data_input(data_input)
        self.aux.set_data_input(data_input)

        for i in range(0, len(data_input.data_inputs)):
            self.constituent_gps[i].set_data_input(data_input.data_inputs[i])

    def copy(self):
        gp = PartitionedGaussianProcess(self.kernel, self.mean_function)
        gp.set_inducing_points(self.inducing_points)
        return gp
