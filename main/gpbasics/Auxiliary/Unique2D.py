from gpbasics import global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf


def tf_unique_2d(x):
    x_shape = x.get_shape()  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
    cond_shape = cond.get_shape()
    cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
    cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return (op)
