import gpbasics.global_parameters as global_param

global_param.ensure_init()

from typing import List

import tensorflow as tf

import gpbasics.Statistics.CovarianceMatrix as cm
import gpbasics.DataHandling.DataInput as di
import gpbasics.MeanFunctionBasics.MeanFunction as mf


class AuxiliaryGpProperties:
    def __init__(self, covariance_matrix: cm.CovarianceMatrix, mean_function: mf.MeanFunction):
        self.covariance_matrix: cm.CovarianceMatrix = covariance_matrix
        self.data_input: di.AbstractDataInput = None
        self.mean_function: mf.MeanFunction = mean_function
        self.detrended_y_train = None
        self.detrended_y_test = None
        self.inv_L_K_dot_K_s = None
        self.posterior_mu = None
        self.posterior_var = None
        self.posterior_sd = None

    def reset(self):
        self.detrended_y_train = None
        self.inv_L_K_dot_K_s = None
        self.posterior_mu = None
        self.posterior_var = None
        self.posterior_sd = None

    def set_data_input(self, data_input: di.AbstractDataInput):
        self.data_input = data_input
        self.reset()

    def get_inverse_cholesky_k_times_k_s(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        pass

    def get_posterior_mu(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        pass

    def get_k_s_diag(self, hyper_param: List[tf.Tensor]):
        pass

    def get_posterior_var(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        pass

    def get_posterior_sd(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        pass


class HolisticAuxiliaryGpProperties(AuxiliaryGpProperties):
    def __init__(self, covariance_matrix: cm.CovarianceMatrix, mean_function: mf.MeanFunction):
        super(HolisticAuxiliaryGpProperties, self).__init__(covariance_matrix, mean_function)

    def get_inverse_cholesky_k_times_k_s(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        if self.data_input is not None:
            if self.inv_L_K_dot_K_s is None:
                self.inv_L_K_dot_K_s = tf.linalg.matmul(self.covariance_matrix.get_L_inv_K(hyper_parameter, noise),
                                                        self.covariance_matrix.get_K_s(hyper_parameter),
                                                        name="inv_L_K_dot_K_s")

            return self.inv_L_K_dot_K_s

        return None

    def get_posterior_mu(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        if self.data_input is not None:
            if self.posterior_mu is None:
                alpha = self.covariance_matrix.get_L_alpha(hyper_parameter, noise)

                K_s = self.covariance_matrix.get_K_s(hyper_parameter)

                posterior_mu = tf.matmul(tf.transpose(K_s), alpha)

                self.posterior_mu = tf.reshape(posterior_mu, [-1, ])

            return self.posterior_mu

        return None

    def get_posterior_var(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        if self.data_input is not None:
            if self.posterior_var is None:
                v = self.get_inverse_cholesky_k_times_k_s(hyper_parameter, noise)
                vtv = tf.matmul(tf.transpose(v), v)
                self.posterior_var = self.covariance_matrix.get_K_ss(hyper_parameter) - vtv

            # For illustrative / graph purposes possibly only the diagonal of that matrix is of interest
            return self.posterior_var

        return None

    def get_posterior_sd(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor):
        if self.data_input is not None:
            if self.posterior_sd is None:
                variance = self.get_posterior_var(hyper_parameter, noise)
                self.posterior_sd = tf.sqrt(variance)

            return self.posterior_sd

        return None


class BlockwiseAuxiliaryGpProperties(HolisticAuxiliaryGpProperties):
    pass
