from typing import List

import gpbasics.global_parameters as global_param

global_param.ensure_init()

from enum import Enum
import tensorflow as tf
import gpbasics.Statistics.CovarianceMatrix as cm
import gpbasics.DataHandling.DataInput as di
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.Metrics.StructuredKernelInterpolation as ski
import gpbasics.Auxiliary.LinearConjugateGradients as lcg
from gpbasics.Statistics.Nystroem_K import NystroemMatrix


class MetricType(Enum):
    LL = 1
    MSE = 5
    BIC = 6
    blockwise_LL = 10
    blockwise_MSE = 50
    blockwise_BIC = 60


class AbstractMetric:
    # metric has to be given in a form having: optimum = minimum
    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices=None) -> tf.Tensor:
        pass

    def get_gradients(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, reset: bool = True) -> tf.Tensor:
        pass


class Metric(AbstractMetric):
    def get_covariance_matrix(self, hyper_param):
        pass

    def get_alpha(self, hyper_param, y):
        pass

    def get_log_determinant(self, hyper_param):
        pass

    def __init__(self, data_input: di.AbstractDataInput, covariance_matrix: cm.CovarianceMatrix, metric_type: MetricType,
                 local_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 subset_size: int = None):
        self.covariance_matrix: cm.CovarianceMatrix = covariance_matrix
        self.local_approx = local_approx
        self.numerical_matrix_handling = numerical_matrix_handling

        self.subset_size = subset_size

        if self.local_approx is not mht.MatrixApproximations.NONE and self.subset_size is None:
            self.subset_size = int(data_input.n_train * global_param.p_nystroem_ratio)

        if self.subset_size is not None and self.subset_size >= data_input.n_train:
            self.local_approx = mht.MatrixApproximations.NONE

        self.data_input: di.DataInput
        if isinstance(self.local_approx, mht.SubsetOfDataApproaches):
            if self.subset_size < data_input.n_train:
                self.data_input = data_input.get_subset(
                    subset_size=self.subset_size, subset_of_data_approach=self.local_approx)
            else:
                self.data_input = data_input
        else:
            self.data_input = data_input

        if self.local_approx is not mht.MatrixApproximations.NONE:
            self.data_input.n_inducting_train = self.subset_size
            self.data_input.n_inducting_test = self.data_input.n_test / self.data_input.n_train * self.subset_size

        self.covariance_matrix.set_data_input(self.data_input)
        self.type: MetricType = metric_type

        if self.local_approx is mht.MatrixApproximations.SKC_LOWER_BOUND or \
                self.local_approx is mht.MatrixApproximations.BASIC_NYSTROEM or \
                self.local_approx is mht.MatrixApproximations.SKC_UPPER_BOUND:
            self.nystroem_matrix = self.get_nystroem_handler()

        if self.numerical_matrix_handling is mht.NumericalMatrixHandlingType.PSEUDO_INVERSE:
            self.get_alpha = self.get_alpha_pseudo_inverse
            self.get_log_determinant = self.get_log_determinant_slodget
        elif self.numerical_matrix_handling is mht.NumericalMatrixHandlingType.CHOLESKY_BASED:
            self.get_alpha = self.get_alpha_cholesky
            self.get_log_determinant = self.get_log_determinant_cholesky
        elif self.numerical_matrix_handling is mht.NumericalMatrixHandlingType.LINEAR_CONJUGATE_GRADIENT:
            self.get_alpha = self.get_alpha_lcg
            self.get_log_determinant = self.get_log_determinant_slodget
        else:
            self.get_alpha = self.get_alpha_strict_inverse
            self.get_log_determinant = self.get_log_determinant_slodget

        if self.local_approx is mht.MatrixApproximations.SKC_UPPER_BOUND:
            self.get_covariance_matrix = self.get_default_covariance_matrix
            self.get_log_determinant = self.get_log_determinant_nystroem
        elif self.local_approx is mht.MatrixApproximations.SKC_LOWER_BOUND:
            self.get_covariance_matrix = self.get_nystroem_matrix
            self.get_log_determinant = self.get_log_determinant_nystroem
        elif self.local_approx is mht.MatrixApproximations.BASIC_NYSTROEM:
            self.get_covariance_matrix = self.get_nystroem_matrix
            self.get_log_determinant = self.get_log_determinant_nystroem
        elif self.local_approx is mht.MatrixApproximations.SKI:
            self.get_covariance_matrix = self.get_ski_matrix
        else:
            self.get_covariance_matrix = self.get_default_covariance_matrix

        self.last_covariance_matrix = None

    def get_nystroem_handler(self) -> NystroemMatrix:
        nyk = NystroemMatrix(self.covariance_matrix)
        nyk.set_data_input(self.data_input)
        return nyk

    def get_nystroem_matrix(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        if self.last_covariance_matrix is None:
            self.last_covariance_matrix = self.nystroem_matrix.get_K_approx_noised(hyper_parameter, noise, indices)
        return self.last_covariance_matrix

    def get_ski_matrix(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        if self.last_covariance_matrix is None:
            self.last_covariance_matrix = \
                ski.get_ski_matrix(hyper_parameter, self.data_input, self.covariance_matrix.kernel, noise)
        return self.last_covariance_matrix

    def get_default_covariance_matrix(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        if self.last_covariance_matrix is None:
            self.last_covariance_matrix = self.covariance_matrix.get_K_noised(hyper_parameter, noise)
        return self.last_covariance_matrix

    def get_alpha_strict_inverse(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, y, indices):
        return tf.matmul(tf.linalg.inv(self.get_covariance_matrix(hyper_parameter, noise, indices)), y)

    def get_alpha_pseudo_inverse(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, y, indices):
        return tf.matmul(tf.linalg.pinv(self.get_covariance_matrix(hyper_parameter, noise, indices)), y)

    def get_alpha_cholesky(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, y, indices):
        return self.covariance_matrix.get_L_alpha(hyper_parameter, noise)

    def get_alpha_lcg(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, y, indices):
        x0 = tf.zeros_like(y, dtype=global_param.p_dtype)
        alpha: tf.Tensor = lcg.linear_cg(self.get_covariance_matrix(hyper_parameter, noise, indices), y, x0)
        return alpha

    def get_log_determinant_slodget(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        return tf.linalg.slogdet(self.get_covariance_matrix(hyper_parameter, noise, indices))[1]

    def get_log_determinant_nystroem(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        return self.nystroem_matrix.get_K_approx_det(hyper_parameter, noise, indices)

    def get_log_determinant_cholesky(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        return 2 * tf.reduce_sum(tf.math.log(
            tf.linalg.diag_part(self.covariance_matrix.get_L_K(hyper_parameter, noise))))
