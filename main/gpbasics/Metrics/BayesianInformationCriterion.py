from typing import List

import tensorflow as tf

from gpbasics import global_parameters as global_param
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpbasics.DataHandling import DataInput as di
from gpbasics.Metrics.LogLikelihood import AbstractLogLikelihood, BlockwiseLogLikelihood
from gpbasics.Metrics.Metrics import Metric, MetricType, AbstractMetric
from gpbasics.Statistics import CovarianceMatrix as cm
from gpbasics.Statistics import GaussianProcess as gp


class AbstractBIC(Metric):
    pass


class BIC(AbstractBIC):
    def __init__(self, data_input: di.DataInput, covariance_matrix: cm.CovarianceMatrix,
                 log_likelihood: AbstractLogLikelihood):
        super(BIC, self).__init__(data_input, covariance_matrix, MetricType.BIC, log_likelihood.local_approx,
                                  log_likelihood.numerical_matrix_handling, log_likelihood.subset_size)
        self.log_likelihood: AbstractLogLikelihood = log_likelihood

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None, reset: bool = True) -> tf.Tensor:
        with tf.name_scope("BIC"):
            # - 2 * log(p(D | M) + |M| * log n = -2 * log_likelihood + |M| * log n
            log_likelihood: tf.Tensor = \
                tf.math.multiply(tf.cast(-1, dtype=global_param.p_dtype),
                                 self.log_likelihood.get_metric(hyper_parameter, noise, indices, reset))

            bic_1: tf.Tensor = tf.math.multiply(tf.cast(-2, dtype=global_param.p_dtype), log_likelihood)
            bic_2: tf.Tensor = tf.math.multiply(tf.cast(self.covariance_matrix.kernel.get_number_of_hyper_parameter(),
                                                        dtype=global_param.p_dtype),
                                                tf.math.log(
                                                    tf.cast(self.data_input.n_train, dtype=global_param.p_dtype)))

            # Penalty (bic_2) is added: smaller BIC = better
            return tf.math.add(bic_1, bic_2)


class BlockwiseBIC(AbstractMetric):
    def __init__(self, _gp: gp.BlockwiseGaussianProcess, local_approx: mht.GlobalApproximationsType,
                 numerical_matrix_handling: mht.NumericalMatrixHandlingType, subset_size: int = None):
        super(BlockwiseBIC, self).__init__()
        self.local_approx: mht.GlobalApproximationsType = local_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size
        self._gp: gp.BlockwiseGaussianProcess = _gp

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None) -> tf.Tensor:
        blockwise_ll: BlockwiseLogLikelihood = BlockwiseLogLikelihood(
            self._gp, self.local_approx, self.numerical_matrix_handling, self.subset_size)

        log_likelihood: tf.Tensor = tf.math.multiply(tf.cast(-1, dtype=global_param.p_dtype),
                                                     blockwise_ll.get_metric(hyper_parameter, noise, indices))

        bic_1: tf.Tensor = tf.math.multiply(tf.cast(-2, dtype=global_param.p_dtype), log_likelihood)
        bic_2: tf.Tensor = \
            tf.math.multiply(tf.cast(self._gp.covariance_matrix.kernel.get_number_of_hyper_parameter(),
                                     dtype=global_param.p_dtype),
                             tf.math.log(tf.cast(self._gp.data_input.n_train, dtype=global_param.p_dtype)))
        return tf.math.add(bic_1, bic_2)
