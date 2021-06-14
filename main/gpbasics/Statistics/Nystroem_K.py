from typing import List

import tensorflow as tf

import gpbasics.Statistics.CovarianceMatrix as cm
import gpbasics.DataHandling.DataInput as di
import gpbasics.global_parameters as global_param
global_param.ensure_init()


class NystroemMatrix:
    def __init__(self, covariance_matrix: cm.CovarianceMatrix):
        self.data_input: di.DataInput = None
        self.covariance_matrix = covariance_matrix
        self.Knm = None
        self.Kmm = None
        self.Kmm_pseudo_inv = None
        self.K_approx = None
        self.K_approx_noised = None
        self.K_approx_inv = None
        self.K_approx_det = None

    def reset(self):
        self.Knm = None
        self.Kmm = None
        self.Kmm_pseudo_inv = None
        self.K_approx = None
        self.K_approx_noised = None
        self.K_approx_inv = None
        self.K_approx_det = None

    def set_data_input(self, data_input: di.AbstractDataInput):
        self.data_input = data_input
        self.reset()

    def get_Knm(self, hyper_param, indices):
        with tf.name_scope("K_nm"):
            self.Knm = self.covariance_matrix.kernel.get_tf_tensor(
                hyper_param, self.data_input.data_x_train, indices)

        return self.Knm

    def get_Kmm(self, hyper_param, indices):
        with tf.name_scope("K_mm"):
            self.Kmm = self.covariance_matrix.kernel.get_tf_tensor(hyper_param, indices, indices)

        return self.Kmm

    def get_Kmm_pseudo_inv(self, hyper_parameter: List[tf.Tensor], indices):
        Kmm = self.get_Kmm(hyper_parameter, indices)

        with tf.name_scope("K_mm_pseudo_inv"):
            self.Kmm_pseudo_inv = tf.linalg.pinv(Kmm, name="inner inverse")

        return self.Kmm_pseudo_inv

    def get_K_approx(self, hyper_parameter: List[tf.Tensor], indices):
        Knm = self.get_Knm(hyper_parameter, indices)
        Kmm_pseudo_inv = self.get_Kmm_pseudo_inv(hyper_parameter, indices)

        with tf.name_scope("K_hat"):
            self.K_approx = tf.matmul(tf.matmul(Knm, Kmm_pseudo_inv), Knm, transpose_b=True)

        return self.K_approx

    def get_K_approx_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        self.get_K_approx(hyper_parameter, indices)
        noise_matrix = tf.eye(self.data_input.n_train, dtype=global_param.p_dtype) * noise
        self.K_approx_noised = self.K_approx + noise_matrix

        return self.K_approx_noised

    def get_K_approx_inv(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        if self.K_approx_inv is None:
            Knm = self.get_Knm(hyper_parameter, indices)
            Knm_t = tf.transpose(Knm)
            Kmm_inv = self.get_Kmm_pseudo_inv(hyper_parameter, indices)

            m = self.data_input.n_inducting_train

            dot_aux = tf.tensordot(Kmm_inv, Knm_t, axes=1)
            Im = tf.cast(noise * tf.eye(m, dtype=global_param.p_dtype), dtype=global_param.p_dtype)
            inner = tf.linalg.pinv(Im + tf.tensordot(dot_aux, Knm, axes=1), name="inner inverse")

            sg_frac = tf.divide(tf.cast(1, dtype=global_param.p_dtype), noise)

            self.K_approx_inv = sg_frac * (tf.eye(self.data_input.n_train, dtype=global_param.p_dtype) -
                                           tf.tensordot(Knm, tf.tensordot(inner, dot_aux, axes=1), axes=1))

        return self.K_approx_inv

    def get_K_approx_det(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor, indices):
        if self.K_approx_det is None:
            Kmm_inv = self.get_Kmm_pseudo_inv(hyper_parameter, indices)
            Knm = self.get_Knm(hyper_parameter, indices)

            m = self.data_input.n_inducting_train
            phi_t = tf.tensordot(Knm, Kmm_inv, axes=1)
            phi = tf.transpose(Knm)

            K_approx_det_1 = tf.cast(self.data_input.n_train - m, dtype=global_param.p_dtype) * tf.math.log(noise)
            Im = tf.eye(m, dtype=global_param.p_dtype) * noise
            to_be_log_det = Im + tf.matmul(phi, phi_t)
            K_approx_det_2 = tf.linalg.slogdet(to_be_log_det)[1]

            self.K_approx_det = K_approx_det_1 + K_approx_det_2

        return self.K_approx_det
