from typing import List

import gpbasics.global_parameters as global_param
global_param.ensure_init()

import tensorflow as tf
from gpbasics.DataHandling import DataInput as di
from gpbasics.KernelBasics import Kernel as k
import gpbasics.Auxiliary.Distances as dist
import numpy as np

def get_ski_matrix(hyper_parameter: List[tf.Tensor], data_input: di.DataInput, kernel: k.Kernel, noise: tf.Tensor):
    if noise is None or noise.shape != []:
        raise Exception("SKI: Invalid noise")

    indices = tf.constant(
        np.linspace(start=0, stop=data_input.n_train, num=data_input.n_inducting_train, endpoint=False, dtype=int),
        dtype=tf.int64, shape=[data_input.n_inducting_train, ])
    K_mm = kernel.get_tf_tensor(hyper_parameter, data_input.get_inducting_x_train(indices),
                                data_input.get_inducting_x_train(indices))

    w_matrix = get_weight_matrix(data_input)

    K_ski = tf.matmul(tf.matmul(w_matrix, K_mm), tf.transpose(w_matrix))

    k_ski_noised = K_ski + tf.eye(data_input.n_train, dtype=global_param.p_dtype) * noise

    return k_ski_noised


def get_weight_matrix(data_input: di.DataInput) -> tf.Tensor:
    distances = dist.euclidian_distance(data_input.data_x_train, data_input.inducting_x_train)

    reduce_min_1 = tf.reshape(tf.reduce_min(distances, axis=1), [-1, 1])
    min_1_condition = distances == reduce_min_1

    mask_1_distances = tf.reduce_max(distances) * tf.cast(min_1_condition, dtype=tf.float64)

    distances_masked = distances + mask_1_distances

    reduce_min_2 = tf.reshape(tf.reduce_min(distances_masked, axis=1), [-1, 1])
    min_2_condition = distances_masked == reduce_min_2

    weight_i = 1 - reduce_min_1 / (reduce_min_1 + reduce_min_2)

    w_matrix = tf.zeros_like(distances) + weight_i * tf.cast(min_1_condition, dtype=tf.float64) + \
               (1 - weight_i) * tf.cast(min_2_condition, dtype=tf.float64)

    return w_matrix


def get_approx_logdet(K_mm, n, m, noise: tf.Tensor):
    if noise is None or noise.shape != []:
        raise Exception("SKI: Invalid noise")

    eigen_values = tf.cast(tf.linalg.eigvals(K_mm), dtype=tf.float64)

    m_eigen_values = (n / m) * eigen_values
    jittered_eig_vals = m_eigen_values + noise
    result = (n / m) * tf.reduce_sum(tf.math.log(jittered_eig_vals))

    return result
