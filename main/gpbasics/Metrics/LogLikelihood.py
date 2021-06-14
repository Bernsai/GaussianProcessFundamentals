from typing import List, cast

import numpy as np
import tensorflow as tf

import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.global_parameters as global_param
from gpbasics.DataHandling import DataInput as di
from gpbasics.KernelBasics import Kernel as k
from gpbasics.Metrics.Metrics import Metric, MetricType, AbstractMetric
from gpbasics.Statistics import CovarianceMatrix as cm
from gpbasics.Statistics import GaussianProcess as gp


class AbstractLogLikelihood(Metric):
    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None, reset: bool = True) \
            -> tf.Tensor:
        pass


class LogLikelihood(AbstractLogLikelihood):
    def __init__(self, data_input: di.AbstractDataInput, covariance_matrix: cm.CovarianceMatrix,
                 local_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 subset_size: int = None):
        super(AbstractLogLikelihood, self).__init__(data_input, covariance_matrix, MetricType.LL, local_approx,
                                                    numerical_matrix_handling, subset_size)
        if self.local_approx is mht.MatrixApproximations.SKC_UPPER_BOUND:
            raise Exception("SKC Upper Bound cannot be handled via default likelihood class")

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None, reset: bool = True) -> tf.Tensor:
        if reset:
            self.covariance_matrix.reset()
            self.last_covariance_matrix = None
        with tf.name_scope("LL"):
            y = self.data_input.get_detrended_y_train()

            alpha: tf.Tensor = self.get_alpha(hyper_parameter, noise, y, indices)

            ll_data_fit: tf.Tensor = -0.5 * tf.matmul(tf.linalg.matrix_transpose(y), alpha)

            ll_complexity_penalty: tf.Tensor = tf.cast(-0.5, dtype=global_param.p_dtype) * \
                                               self.get_log_determinant(hyper_parameter, noise, indices)

            log_2_pi: tf.Tensor = tf.math.log(tf.math.multiply(tf.cast(np.pi, dtype=global_param.p_dtype), 2))
            ll_norm_constant: tf.Tensor = tf.math.multiply(tf.cast(-0.5, dtype=global_param.p_dtype),
                                                           tf.math.multiply(self.data_input.n_train, log_2_pi))

            # since only norm constant is always a scalar value (even for batch input)
            log_likelihood: tf.Tensor = tf.math.add(ll_data_fit, ll_complexity_penalty) + ll_norm_constant

            if self.local_approx is mht.MatrixApproximations.SKC_LOWER_BOUND:
                correction_term_frac: tf.Tensor = tf.divide(1, tf.multiply(tf.cast(2, dtype=global_param.p_dtype),
                                                                           global_param.p_cov_matrix_jitter))

                # Correction term is VERY important! cf. Titsias.2009
                correction_term: tf.Tensor = correction_term_frac * tf.linalg.trace(
                    self.get_covariance_matrix(hyper_parameter, noise, indices) -
                    self.covariance_matrix.get_K(hyper_parameter))

                log_likelihood = log_likelihood - correction_term

            if len(self.data_input.data_x_train.shape) == 3:
                log_likelihood = global_param.p_batch_metric_aggregator(log_likelihood)

            return -log_likelihood


class BlockwiseLogLikelihood(AbstractMetric):
    def __init__(self, _gp: gp.BlockwiseGaussianProcess, local_approx: mht.GlobalApproximationsType,
                 numerical_matrix_handling: mht.NumericalMatrixHandlingType, subset_size: int = None):
        super(AbstractMetric, self).__init__()
        self.local_approx: mht.GlobalApproximationsType = local_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size
        self._gp: gp.BlockwiseGaussianProcess = _gp

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices, reset: bool = True) -> tf.Tensor:
        ll_sum: List[tf.Tensor] = []
        index: int
        if not hasattr(self._gp, 'change_point_positions'):
            index = 0
        else:
            index = len(self._gp.covariance_matrix.kernel.change_point_positions)
        i = 0
        for idx, sub_gp in enumerate(self._gp.constituent_gps):
            sub_ll: LogLikelihood = cast(LogLikelihood, LogLikelihood(
                sub_gp.data_input, sub_gp.covariance_matrix, self.local_approx,
                self.numerical_matrix_handling, self.subset_size))

            kernel: k.Kernel = sub_gp.covariance_matrix.kernel

            slice_hyper_param: List[tf.Tensor] = hyper_parameter[index: index + kernel.get_number_of_hyper_parameter()]

            summand: tf.Tensor = sub_ll.get_metric(slice_hyper_param, noise, indices)
            ll_sum.append(summand)

            index += kernel.get_number_of_hyper_parameter()

            sub_gp.covariance_matrix.reset()
            sub_gp.aux.reset()

            i += 1

        return tf.add_n(ll_sum)
