import tensorflow as tf


@tf.function
def euclidian_distance(a: tf.Tensor, b: tf.Tensor):
    return tf.sqrt(tf.reduce_sum(a*a, axis=-1, keepdims=True) - 2 * tf.matmul(a, tf.linalg.matrix_transpose(b)) +
                   tf.linalg.matrix_transpose(tf.reduce_sum(b*b, axis=-1, keepdims=True)))


@tf.function
def manhattan_distance(a: tf.Tensor, b: tf.Tensor):
    return tf.reduce_sum(tf.abs(tf.expand_dims(a, -2) - tf.expand_dims(b, -3)), -1)
