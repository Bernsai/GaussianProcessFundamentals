from typing import List, cast

import numpy as np
import tensorflow as tf

import gpbasics.MeanFunctionBasics.MeanFunction as mf
import gpbasics.global_parameters as global_param

global_param.ensure_init()


class BaseMeanFunction(mf.MeanFunction):
    def __init__(self, manifestation, input_dimensionality: int):
        super(BaseMeanFunction, self).__init__(
            mf.MeanFunctionType.BASE_MEAN_FUNCTION, manifestation, input_dimensionality)
        if self.manifestation.value > 199:
            print("Invalid manifestation for BaseKernel: ", manifestation)

    def get_number_base_mean_function(self) -> int:
        return 1

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        assert len(
            last_hyper_parameter) == self.get_number_of_hyper_parameter(), "Wrong size/shape of given 'last_hyper_param'"
        self.last_hyper_parameter = last_hyper_parameter

    def get_number_of_hyper_parameter(self) -> int:
        return len(self.get_default_hyper_parameter())

    def get_string_representation(self) -> str:
        return self.manifestation.name

    def get_string_representation_weight(self) -> int:
        return self.manifestation.value - 100


class ConstantMeanFunction(BaseMeanFunction):
    def __init__(self, input_dimensionality: int):
        super(ConstantMeanFunction, self).__init__(mf.MeanFunctionManifestation.C, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray) -> tf.Tensor:
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("ConstantMeanFunction"):
            zeroes: tf.Tensor = tf.zeros(shape=[x_vector.shape[0], ], dtype=global_param.p_dtype)
            result: tf.Tensor = tf.add(zeroes, hyper_parameter[0], name="ConstantMeanFunction")

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        hyp_dims = self.get_hyper_parameter_dimensionalities()
        return [tf.Variable(0.01, dtype=global_param.p_dtype, shape=hyp_dims[0])]

    def deepcopy(self):
        copied_constant_mean_function = ConstantMeanFunction(self.input_dimensionality)
        copied_constant_mean_function.set_last_hyper_parameter(self.last_hyper_parameter)
        return copied_constant_mean_function

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return [[]]


class ZeroMeanFunction(ConstantMeanFunction):
    def __init__(self, input_dimensionality: int):
        super(ZeroMeanFunction, self).__init__(input_dimensionality)

    def get_string_representation(self) -> str:
        return "ZERO_MEAN"

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        hyp_dims = self.get_hyper_parameter_dimensionalities()
        return [tf.Variable(0, dtype=global_param.p_dtype, shape=hyp_dims[0])]

    def deepcopy(self):
        copied_zero_mean_function = ZeroMeanFunction(self.input_dimensionality)
        return copied_zero_mean_function


class LinearMeanFunction(BaseMeanFunction):
    def __init__(self, input_dimensionality: int):
        super(LinearMeanFunction, self).__init__(mf.MeanFunctionManifestation.LIN, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray) -> tf.Tensor:
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("LinearMeanFunction"):
            sloped: tf.Tensor = tf.multiply(x_vector, hyper_parameter[0], name="sloped")
            result: tf.Tensor = tf.reduce_sum(sloped, axis=1)

        self.last_hyper_parameter = hyper_parameter

        assert int(x_vector.shape[0]) == int(result.shape[0]), "Shape mismatch for LinearMeanFunction"

        return result

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        hyp_dims = self.get_hyper_parameter_dimensionalities()
        dim_scale: float = 1 / self.input_dimensionality
        return cast(List[tf.Tensor], [tf.Variable(
            tf.fill(dims=hyp_dims[0], value=tf.Variable(dim_scale, dtype=global_param.p_dtype)))])

    def deepcopy(self):
        copied_linear_mean_function = LinearMeanFunction(self.input_dimensionality)
        copied_linear_mean_function.set_last_hyper_parameter(self.last_hyper_parameter)
        return copied_linear_mean_function

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return [[self.input_dimensionality, ]]


class ExponentialMeanFunction(BaseMeanFunction):
    def __init__(self, input_dimensionality: int):
        super(ExponentialMeanFunction, self).__init__(mf.MeanFunctionManifestation.EXP, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray) -> tf.Tensor:
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("ExponentialMeanFunction"):
            scaled: tf.Tensor = tf.multiply(hyper_parameter[0], x_vector)

            shifted: tf.Tensor = tf.subtract(scaled, hyper_parameter[1])

            potency: tf.Tensor = tf.pow(hyper_parameter[2], tf.reduce_sum(shifted, axis=1))

        self.last_hyper_parameter = hyper_parameter

        return potency

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        hyp_dims = self.get_hyper_parameter_dimensionalities()
        scale = tf.Variable(tf.fill(dims=hyp_dims[0],
                                    value=tf.Variable(1, dtype=global_param.p_dtype)))

        shift = tf.Variable(tf.fill(dims=hyp_dims[1],
                                    value=tf.Variable(0, dtype=global_param.p_dtype)))

        base = tf.Variable(np.e, dtype=global_param.p_dtype, shape=hyp_dims[2])
        return [scale, shift, base]

    def deepcopy(self):
        copied_exponential_mean_function = ExponentialMeanFunction(self.input_dimensionality)
        copied_exponential_mean_function.set_last_hyper_parameter(self.last_hyper_parameter)
        return copied_exponential_mean_function

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return [[self.input_dimensionality, ], [self.input_dimensionality, ], []]


class LogitMeanFunction(BaseMeanFunction):
    def __init__(self, input_dimensionality: int):
        super(LogitMeanFunction, self).__init__(mf.MeanFunctionManifestation.LOGIT, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray) -> tf.Tensor:
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("LogitMeanFunction"):
            steeped: tf.Tensor = tf.multiply(hyper_parameter[0], x_vector)

            shifted: tf.Tensor = tf.reduce_sum(tf.subtract(steeped, hyper_parameter[1]), axis=1)

            denominator: tf.Tensor = tf.add(tf.cast(1, dtype=global_param.p_dtype), tf.exp(shifted))
            fraction: tf.Tensor = tf.divide(hyper_parameter[2], denominator)
            result: tf.Tensor = fraction

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        hyp_dims = self.get_hyper_parameter_dimensionalities()
        steep = tf.Variable(tf.fill(dims=hyp_dims[0],
                                    value=tf.Variable(-1, dtype=global_param.p_dtype)))

        shift = tf.Variable(tf.fill(dims=hyp_dims[1],
                                    value=tf.Variable(0, dtype=global_param.p_dtype)))

        max_value = tf.Variable(1, dtype=global_param.p_dtype, shape=hyp_dims[2])

        return [steep, shift, max_value]

    def deepcopy(self):
        copied_logit_mean_function = LogitMeanFunction(self.input_dimensionality)
        copied_logit_mean_function.set_last_hyper_parameter(self.last_hyper_parameter)
        return copied_logit_mean_function

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return [[self.input_dimensionality, ], [self.input_dimensionality, ], []]
