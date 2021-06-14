from gpbasics import global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf
from enum import Enum


class SimilarityType(Enum):
    LINEAR = 0
    SQRT_LINEAR = 1
    LOG_LINEAR = 2
    RECIPROCAL = 3


def get_similarity_based_distance(distance, similarity_type: SimilarityType):
    if similarity_type is SimilarityType.LINEAR:
        return get_linear_similarity(distance)
    elif similarity_type is SimilarityType.SQRT_LINEAR:
        return get_sqrt_linear_similarity(distance)
    elif similarity_type is SimilarityType.LOG_LINEAR:
        return get_log_linear_similarity(distance)
    elif similarity_type is SimilarityType.RECIPROCAL:
        return get_reciprocal_similarity(distance)
    else:
        raise Exception("SimilarityBased Distance type invalid: %s" % str(similarity_type))


def get_linear_similarity(distance):
    return 1 - distance


def get_sqrt_linear_similarity(distance):
    return tf.sqrt(get_linear_similarity(distance))


def get_log_linear_similarity(distance):
    return tf.math.log(get_linear_similarity(distance))


def get_reciprocal_similarity(distance):
    return tf.divide(1, distance) - 1
