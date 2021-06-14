from typing import List

import tensorflow as tf

from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput
from gpbasics.Metrics import MatrixHandlingTypes as mht
from gpbasics.DataHandling import DataInput as di
from gpbasics.KernelBasics import Kernel as k
from gpbasics.Metrics.Metrics import Metric, MetricType, AbstractMetric
from gpbasics.Statistics import CovarianceMatrix as cm, Auxiliary as ax
from gpbasics.Statistics import GaussianProcess as gp


class AbstractMSE(Metric):
    pass


class MeanSquaredError(AbstractMSE):
    def __init__(self, data_input: di.DataInput, covariance_matrix: cm.CovarianceMatrix,
                 aux_gp: ax.AuxiliaryGpProperties, local_approx: mht.GlobalApproximationsType,
                 numerical_matrix_handling: mht.NumericalMatrixHandlingType, subset_size: int = None):
        super(MeanSquaredError, self).__init__(data_input, covariance_matrix, MetricType.MSE, local_approx,
                                               numerical_matrix_handling, subset_size)
        self.aux_gp: ax.AuxiliaryGpProperties = aux_gp

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None) -> tf.Tensor:
        self.aux_gp.reset()
        self.covariance_matrix.reset()
        post_mu = tf.reshape(self.get_posterior_mu(hyper_parameter, noise, indices), [-1, 1])
        detrended_y_test = self.data_input.get_detrended_y_test()
        return tf.reduce_mean(tf.math.squared_difference(post_mu, detrended_y_test))

    def get_posterior_mu(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None):
        y = self.data_input.get_detrended_y_train()

        alpha = self.get_alpha(hyper_parameter, noise, y, indices)

        K_s = self.covariance_matrix.get_K_s(hyper_parameter)

        posterior_mu = tf.matmul(tf.transpose(K_s), alpha)

        return tf.reshape(posterior_mu, [int(self.data_input.n_test), ])


class BlockwiseMeanSquaredError(AbstractMetric):
    def __init__(self, _gp: gp.BlockwiseGaussianProcess, local_approx: mht.GlobalApproximationsType,
                 numerical_matrix_handling: mht.NumericalMatrixHandlingType, subset_size: int = None):
        super(BlockwiseMeanSquaredError, self).__init__()
        self.local_approx: mht.GlobalApproximationsType = local_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size
        self.aux_gp: ax.AuxiliaryGpProperties = _gp.aux
        self._gp: gp.BlockwiseGaussianProcess = _gp
        self.data_input: AbstractDataInput = _gp.data_input

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None) -> tf.Tensor:
        if not hasattr(self._gp, 'change_point_positions'):
            index = 0
        else:
            index = len(self._gp.covariance_matrix.kernel.change_point_positions)

        post_mus = []
        detrended_y_tests = []
        for sub_gp in self._gp.constituent_gps:
            sub_mse: Metric = MeanSquaredError(sub_gp.data_input, sub_gp.covariance_matrix, sub_gp.aux,
                                               self.local_approx, self.numerical_matrix_handling, self.subset_size)

            assert isinstance(sub_mse, MeanSquaredError)

            kernel: k.Kernel = sub_gp.covariance_matrix.kernel
            slice_hyper_param: List[tf.Tensor] = hyper_parameter[index: index + kernel.get_number_of_hyper_parameter()]
            post_mus.append(
                sub_mse.get_posterior_mu(slice_hyper_param, noise, indices))
            detrended_y_tests.append(
                sub_mse.data_input.get_detrended_y_test()
            )
            index += kernel.get_number_of_hyper_parameter()

        post_mu = tf.reshape(tf.concat(post_mus, axis=0), [-1, 1])
        detrended_y_test = tf.reshape(tf.concat(detrended_y_tests, axis=0), [-1, 1])
        return tf.reduce_mean(tf.math.squared_difference(post_mu, detrended_y_test))
