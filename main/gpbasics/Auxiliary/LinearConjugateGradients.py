from gpbasics import global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf
import logging
import numpy as np

def linear_cg(matrix: tf.Tensor, vector: tf.Tensor, x: tf.Tensor):
    n = int(tf.shape(matrix)[0])
    r = [tf.matmul(matrix, x) - vector]
    p = [-r[0]]
    k = 0

    first_run = True

    while first_run or tf.abs(tf.reduce_max(r[k])) > 1e-2:
        Apk = get_Apk(matrix, p[k])
        alpha_k = tf.reshape(get_alpha_k(r[k], p[k], Apk), [])
        next_x = get_next_x_k(x, alpha_k, p[k])
        if bool(tf.reduce_any(tf.math.is_nan(next_x))):
            return x
        else:
            x = next_x
        r.append(get_next_r_k(r[k], alpha_k, Apk))
        next_beta_k = tf.reshape(get_next_beta_k(r[k + 1], r[k]), [])
        p.append(get_next_p_k(r[k + 1], next_beta_k, p[k]))
        k += 1
        if k - 1 >= 0:
            p[k - 1] = None
            r[k - 1] = None

        first_run = False

        if k % (n / 4) == 0:
            if k > n:
                logging.debug(
                    "Linear Conjugate Gradient cannot be determined. Amount of iterations exceeds n (=%i)." % n)
                break

    return x


def get_Apk(matrix, p_k):
    return tf.matmul(matrix, p_k)


def get_alpha_k(r_k, p_k, Apk):
    numerator = tf.matmul(tf.transpose(r_k), r_k)
    denominator = tf.matmul(tf.transpose(p_k), Apk)

    return numerator / denominator


def get_next_x_k(previous_x_k, alpha_k, p_k):
    return previous_x_k + alpha_k * p_k


def get_next_r_k(previous_r_k, alpha_k, Apk):
    return previous_r_k + alpha_k * Apk


def get_next_beta_k(next_r_k, r_k):
    numerator = tf.matmul(tf.transpose(next_r_k), next_r_k)
    denominator = tf.matmul(tf.transpose(r_k), r_k)

    return numerator / denominator


def get_next_p_k(next_r_k, next_beta_k, p_k):
    return (-1 * next_r_k) + next_beta_k * p_k
