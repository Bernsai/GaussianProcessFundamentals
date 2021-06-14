from gpbasics import global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf


def build_block_matrix_from_non_square_matrices(block_matrices):
    n_train, n_test = 0, 0
    for bm in block_matrices:
        if isinstance(bm, tf.Tensor):
            shape = tf.shape(bm).numpy().tolist()
        else:
            shape = bm

        n_train += shape[0]
        n_test += shape[1]

    block_matrix: tf.Tensor = None
    dead_cols = 0
    dead_rows = 0
    for bm in block_matrices:
        if block_matrix is None:
            if isinstance(bm, tf.Tensor):
                if dead_rows == 0 and dead_cols == 0:
                    block_matrix = bm
                else:
                    if dead_rows > 0 and dead_cols > 0:
                        block_matrix = tf.zeros(shape=[dead_rows, dead_cols], dtype=global_param.p_dtype)
                        block_matrix = build_block_matrix_from_two_non_square_matrices(block_matrix, bm)
                    elif dead_rows > 0:
                        block_matrix = add_dead_row_shift(bm, dead_rows, reverse=True)
                    elif dead_cols > 0:
                        block_matrix = add_dead_row_shift(bm, dead_cols, reverse=True)
                    else:
                        raise Exception("Invalid state!")
                    dead_cols, dead_rows = 0, 0
            elif bm[0] != bm[1]:
                if bm[0] == 0:
                    dead_cols += bm[1]
                elif bm[1] == 0:
                    dead_rows += bm[0]
            else:
                raise Exception("PartitionOperator: One given input vector is empty for current partition. "
                                "Empty partition is first partition!")
        else:
            if isinstance(bm, tf.Tensor):
                assert dead_cols == 0 and dead_rows == 0, \
                    "Dead rows and dead cols handling only necessary for " \
                    "empty partitions before first non-empty partition."
                block_matrix = build_block_matrix_from_two_non_square_matrices(block_matrix, bm)
            elif bm[0] != bm[1]:
                # Handling of empty partitions
                if dead_rows > 0 or dead_cols > 0:
                    # Not only the first but consecutive ones as well are empty
                    if bm[0] == 0:
                        dead_cols += bm[1]
                    elif bm[1] == 0:
                        dead_rows += bm[0]
                else:
                    # Handling of empty partitions preceded by at least one non empty
                    if bm[0] == 0:
                        block_matrix = add_dead_column_shift(block_matrix, bm[1])
                    elif bm[1] == 0:
                        block_matrix = add_dead_row_shift(block_matrix, bm[0])
                    else:
                        raise Exception("PartitionOperator: One given input vector is empty for current partition. "
                                        "Invalid shape given: %s" % str(bm))
    return block_matrix


def add_dead_row_shift(block_matrix, dead_rows: int, reverse: bool = False):
    zeros_lower = tf.zeros(shape=[dead_rows, block_matrix.shape[1]], dtype=global_param.p_dtype)
    to_be_concatenated = [block_matrix, zeros_lower]

    if reverse:
        to_be_concatenated.reverse()

    block_matrix = tf.concat(to_be_concatenated, axis=0)
    return block_matrix


def add_dead_column_shift(block_matrix: tf.Tensor, dead_columns: int, reverse: bool = False):
    zeros_upper = tf.zeros(shape=[block_matrix.shape[0], dead_columns], dtype=global_param.p_dtype)
    to_be_concatenated = [block_matrix, zeros_upper]

    if reverse:
        to_be_concatenated.reverse()

    block_matrix = tf.concat(to_be_concatenated, axis=1)
    return block_matrix


def build_block_matrix_from_two_non_square_matrices(block_matrix_a, block_matrix_b):
    zeros_upper_shape = [block_matrix_a.shape[0], block_matrix_b.shape[1]]
    upper_part = \
        tf.concat([block_matrix_a, tf.zeros(shape=zeros_upper_shape, dtype=global_param.p_dtype)],
                  axis=1)
    zeros_lower_shape = [block_matrix_b.shape[0], block_matrix_a.shape[1]]
    lower_part = tf.concat([tf.zeros(shape=zeros_lower_shape, dtype=global_param.p_dtype), block_matrix_b],
                           axis=1)
    block_matrix_a = tf.concat([upper_part, lower_part], axis=0)
    return block_matrix_a
