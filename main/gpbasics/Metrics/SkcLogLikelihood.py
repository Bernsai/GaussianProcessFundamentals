from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpbasics.global_parameters as global_param
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpbasics.DataHandling import DataInput as di
from gpbasics.Metrics.Metrics import MetricType, Metric
from gpbasics.Statistics import CovarianceMatrix as cm, Nystroem_K as ny


class LogLikelihoodUpperBound(Metric):
    def __init__(self, data_input: di.DataInput, covariance_matrix: cm.CovarianceMatrix,
                 nystroem_k: ny.NystroemMatrix):
        super(LogLikelihoodUpperBound, self).__init__(
            data_input, covariance_matrix, MetricType.LL, local_approx=mht.MatrixApproximations.SKC_UPPER_BOUND,
            numerical_matrix_handling=mht.NumericalMatrixHandlingType.LINEAR_CONJUGATE_GRADIENT)

        self.nyK: ny.NystroemMatrix = nystroem_k
        self.hyper_parameter = None
        self.noise = None
        self.indices = None

    def optimizable(self, alpha) -> tf.Tensor:
        with tf.name_scope("upperLL"):
            y = self.data_input.get_detrended_y_train()
            alpha: tf.Tensor = tf.reshape(alpha, shape=[self.data_input.n_train, 1])
            with tf.name_scope("data_fit"):
                K_noised: tf.Tensor = self.get_covariance_matrix(self.hyper_parameter, self.noise, self.indices)

                ll_data_fit_sum_one: tf.Tensor = \
                    0.5 * tf.matmul(tf.matmul(tf.transpose(alpha), K_noised, name="Matmul_inner"), alpha
                                    , name="Matmul_outer")
                ll_data_fit_sum_two: tf.Tensor = tf.matmul(tf.transpose(alpha), y, name="Matmul_two")

                ll_data_fit: tf.Tensor = tf.reshape(tf.subtract(ll_data_fit_sum_one, ll_data_fit_sum_two), [])

            with tf.name_scope("CompPenalty"):
                K_approx_det: tf.Tensor = self.nyK.get_K_approx_det(self.hyper_parameter, self.noise, self.indices)
                ll_complexity_penalty: tf.Tensor = tf.cast(-0.5, dtype=global_param.p_dtype) * K_approx_det

            with tf.name_scope("NormConstant"):
                log_2_pi: tf.Tensor = tf.math.log(tf.multiply(tf.cast(np.pi, dtype=global_param.p_dtype), 2))
                ll_norm_constant: tf.Tensor = tf.multiply(tf.cast(-0.5, dtype=global_param.p_dtype),
                                                          int(self.data_input.n_train)) * log_2_pi

            upper_bound = tf.add_n([ll_data_fit, ll_complexity_penalty, ll_norm_constant])

        return upper_bound

    def get_metric(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices) -> tf.Tensor:
        self.hyper_parameter = hyper_parameter
        self.noise = noise
        self.indices = indices
        alpha: tf.Variable = tf.Variable(tf.ones(shape=[self.data_input.n_train, ], dtype=global_param.p_dtype))

        def opt():
            result = tfp.math.value_and_gradient(self.optimizable, alpha)
            return result

        sgd_opt: tfp.optimizer.VariationalSGD = tfp.optimizer.VariationalSGD(10, 10)

        sgd_opt.minimize(opt, alpha)

        post_fit_metric = self.optimizable(alpha)

        return post_fit_metric
