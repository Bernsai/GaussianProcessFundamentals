import logging
from typing import List, cast, Tuple

import numpy as np
import tensorflow as tf

import gpbasics.Auxiliary.Distances as aux_dist
import gpbasics.KernelBasics.Kernel as k
import gpbasics.global_parameters as global_param

global_param.ensure_init()


class BaseKernel(k.Kernel):
    def __init__(self, manifestation, input_dimensionality: int):
        super(BaseKernel, self).__init__(k.KernelType.BASE_KERNEL, manifestation, input_dimensionality)

        if self.manifestation.value > 199:
            logging.critical("Invalid manifestation for BaseKernel: ", manifestation)

    def get_number_base_kernels(self) -> int:
        return 1

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        if not isinstance(last_hyper_parameter, list) and not \
                (isinstance(last_hyper_parameter[0], tf.Variable) or isinstance(last_hyper_parameter[0], tf.Tensor)):
            raise Exception("Wrong type for last_hyper_parameter to be set!")
        assert len(last_hyper_parameter) == self.get_number_of_hyper_parameter(), \
            "Invalid hyper_param size: %s" % str(self)
        self.last_hyper_parameter = last_hyper_parameter

    def get_last_hyper_parameter(self):
        return self.last_hyper_parameter

    def get_string_representation_weight(self) -> int:
        return self.manifestation.value - 100

    def get_json(self) -> dict:
        hp = self.get_last_hyper_parameter()

        return {"type": self.get_string_representation(), "hyper_param": [h.numpy().tolist() for h in hp]}

    def get_string_representation(self) -> str:
        return self.manifestation.name

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        pass

    def get_number_of_child_nodes(self) -> int:
        return 1


class ConstantKernel(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(ConstantKernel, self).__init__(k.KernelManifestation.C, input_dimensionality)
        raise Exception("Not up to date. Implementation for ConstantKernel is not available.")

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), \
            "Invalid hyper_parameter size: " + str(self)

        with tf.name_scope("ConstantKernel"):
            ones = tf.ones([x_vector.shape[0], x_vector_.shape[0]], global_param.p_dtype, name="ones")
            ones = tf.multiply(ones, hyper_parameter[0], name="ConstantKernel")

        self.last_hyper_parameter = hyper_parameter

        return ones

    def get_number_of_hyper_parameter(self) -> int:
        return 1

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        return cast(List[tf.Tensor],
                    [tf.fill(dims=hyp_dim[0], value=0.01, dtype=global_param.p_dtype)])

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return [[self.input_dimensionality, ]]

    def deepcopy(self):
        copied_constant_kernel: ConstantKernel = ConstantKernel(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_constant_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_constant_kernel.set_noise(self.noise)

        return copied_constant_kernel

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        return [representation + "_c"]

    def type_compare_to(self, other):
        return isinstance(other, ConstantKernel)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(ConstantKernel, self).get_last_hyper_parameter()
        return result


class LinearKernel(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(LinearKernel, self).__init__(k.KernelManifestation.LIN, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("LinearKernel"):
            x_off = tf.math.subtract(x_vector, hyper_parameter[0], name="x_minus_c")
            x__off = tf.math.subtract(x_vector_, hyper_parameter[0], name="x-_minus_c")

            if len(x_vector.shape) == 3:
                result = tf.matmul(x_off, tf.transpose(x__off, perm=[0, 2, 1]))
            elif len(x_vector.shape) == 4:
                result = tf.matmul(x_off, tf.transpose(x__off, perm=[0, 1, 3, 2]))
            else:
                result = tf.matmul(x_off, tf.transpose(x__off))

            if global_param.p_scaled_base_kernel:
                result = tf.multiply(hyper_parameter[1], result)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        c_hypdim = self.get_hyper_parameter_dimensionalities()
        c_lower_bound = \
            tf.reshape(tf.repeat(
                tf.constant(-np.inf, dtype=global_param.p_dtype), repeats=self.input_dimensionality), c_hypdim[0])

        c_upper_bound = \
            tf.reshape(tf.repeat(
                tf.constant(np.inf, dtype=global_param.p_dtype), repeats=self.input_dimensionality), c_hypdim[0])

        if self.get_number_of_hyper_parameter() == 2:
            scale_lower_bound = tf.cast(global_param.p_cov_matrix_jitter * 100, dtype=global_param.p_dtype)
            scale_upper_bound = tf.constant(np.inf, dtype=global_param.p_dtype)
            return [(c_lower_bound, c_upper_bound), (scale_lower_bound, scale_upper_bound)]
        else:
            return [(c_lower_bound, c_upper_bound)]

    def get_number_of_hyper_parameter(self) -> int:
        number_hyper_param = 1

        if global_param.p_scaled_base_kernel:
            number_hyper_param += 1

        return number_hyper_param

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        if from_distribution:
            return self.get_default_hyper_parameter_distribution(xrange, n)
        else:
            return self.get_default_hyper_parameter_fixed(xrange, n)

    def get_default_hyper_parameter_fixed(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        default_hyp = cast(List[tf.Tensor], [tf.Variable(
            tf.fill(dims=hyp_dim[0], value=tf.Variable(0.01, dtype=global_param.p_dtype)))])

        if global_param.p_scaled_base_kernel:
            default_hyp.append(cast(tf.Tensor, tf.Variable(0.1, shape=hyp_dim[1], dtype=global_param.p_dtype)))

        return default_hyp

    def get_default_hyper_parameter_distribution(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        default_hyp: List[tf.Tensor] = []
        base = tf.constant(xrange, dtype=global_param.p_dtype)
        x_min = tf.reshape(tf.gather(base, [0], axis=1), [-1, ])
        x_max = tf.reshape(tf.gather(base, [1], axis=1), [-1, ])
        x_range = x_max - x_min

        offset = tf.Variable(tf.random.uniform(
            hyp_dim[0], minval=tf.reduce_min(x_min - x_range), maxval=tf.reduce_max(x_max + x_range),
            dtype=global_param.p_dtype))

        default_hyp.append(offset)

        if global_param.p_scaled_base_kernel:
            scale = tf.Variable(tf.abs(tf.random.normal(hyp_dim[1], mean=0.1, stddev=0.2, dtype=global_param.p_dtype)))
            default_hyp.append(scale)

        return default_hyp

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        base = tf.constant(xrange, dtype=global_param.p_dtype)
        x_min = tf.reshape(tf.gather(base, [0], axis=1), [-1, ])
        x_max = tf.reshape(tf.gather(base, [1], axis=1), [-1, ])
        x_range = x_max - x_min

        result.append({"shape": hyp_dim[0], "minval": x_min - x_range, "maxval": x_max + x_range,
                       "type": "random_uniform"})

        if global_param.p_scaled_base_kernel:
            result.append({"shape": hyp_dim[1], "mean": 0.1, "stddev": 0.2, "type": "random_normal"})

        return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        dimensionality = [[self.input_dimensionality, ]]

        if global_param.p_scaled_base_kernel:
            dimensionality.append([])

        return dimensionality

    def deepcopy(self):
        copied_linear_kernel = LinearKernel(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_linear_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_linear_kernel.set_noise(self.noise)

        return copied_linear_kernel

    @staticmethod
    @tf.function
    def get_c_derivative_matrix(hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> tf.Tensor:
        diff = aux_dist.manhattan_distance(x_vector, x_vector_)
        return tf.add(diff, tf.multiply(tf.cast(2, global_param.p_dtype), hyper_parameter[0]))

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        return [self.get_c_derivative_matrix(hyper_parameter, x_vector, x_vector_)]

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        names = [representation + "_c"]

        if global_param.p_scaled_base_kernel:
            names.append(representation + "_sg")

        return names

    def type_compare_to(self, other):
        return isinstance(other, LinearKernel)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(LinearKernel, self).get_last_hyper_parameter()
        if scaling_x_param is None:
            return result

        scaled_hyper_param = [result[0] * scaling_x_param[1] + scaling_x_param[0]]

        if global_param.p_scaled_base_kernel:
            scaled_hyper_param.append(result[1])

        return scaled_hyper_param


class SquaredExponentialKernel(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(SquaredExponentialKernel, self).__init__(k.KernelManifestation.SE, input_dimensionality)
        self.latest_cov_mat: tf.Tensor = None

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        with tf.name_scope("SquaredExponentialKernel"):
            dist: tf.Tensor = aux_dist.euclidian_distance(x_vector, x_vector_)
            squared_distance: tf.Tensor = tf.square(dist, name="squared_distance")

            result: tf.Tensor = \
                tf.math.exp(-0.5 * tf.math.divide(squared_distance, tf.square(hyper_parameter[0], name="l_squared")),
                            name="SquaredExponentialKernel")

            if global_param.p_scaled_base_kernel:
                result = tf.multiply(hyper_parameter[1], result)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        range_length = xrange[0][1] - xrange[0][0]
        l_lower_bound = tf.cast((5 * range_length / n), global_param.p_dtype)
        l_upper_bound = tf.cast((range_length / 3), global_param.p_dtype)

        if self.get_number_of_hyper_parameter() == 2:
            scale_lower_bound = tf.cast(global_param.p_cov_matrix_jitter * 100, dtype=global_param.p_dtype)
            scale_upper_bound = tf.constant(np.inf, dtype=global_param.p_dtype)
            return [(l_lower_bound, l_upper_bound), (scale_lower_bound, scale_upper_bound)]
        else:
            return [(l_lower_bound, l_upper_bound)]

    def get_number_of_hyper_parameter(self) -> int:
        number_hyper_param = 1

        if global_param.p_scaled_base_kernel:
            number_hyper_param += 1

        return number_hyper_param

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        if from_distribution:
            return self.get_default_hyper_parameter_distribution(xrange, n)
        else:
            return self.get_default_hyper_parameter_fixed(xrange, n)

    def get_default_hyper_parameter_fixed(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        width = xrange[0][1] - xrange[0][0]
        initial_value = width / 10
        length_scale = cast(tf.Tensor, tf.Variable(initial_value, dtype=global_param.p_dtype, shape=hyp_dim[0]))

        if global_param.p_scaled_base_kernel:
            return [length_scale, cast(tf.Tensor, tf.Variable(0.1, dtype=global_param.p_dtype, shape=hyp_dim[1]))]

        return [length_scale]

    def get_default_hyper_parameter_distribution(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        default_hyp: List[tf.Tensor] = []

        width = xrange[0][1] - xrange[0][0]

        length_scale = tf.Variable(
            tf.abs(tf.random.normal(hyp_dim[0], mean=width / 10, stddev=0.2, dtype=global_param.p_dtype)))

        default_hyp.append(length_scale)

        if global_param.p_scaled_base_kernel:
            scale = tf.Variable(tf.abs(tf.random.normal(hyp_dim[1], mean=0.1, stddev=0.2, dtype=global_param.p_dtype)))
            default_hyp.append(scale)

        return default_hyp

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        width = xrange[0][1] - xrange[0][0]

        result.append({"shape": hyp_dim[0], "mean": width / 10, "stddev": 0.2, "type": "random_normal"})

        if global_param.p_scaled_base_kernel:
            result.append({"shape": hyp_dim[1], "mean": 0.1, "stddev": 0.2, "type": "random_normal"})

        return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        dimensionality = [[]]

        if global_param.p_scaled_base_kernel:
            dimensionality.append([])

        return dimensionality

    def deepcopy(self):
        copied_se_kernel = SquaredExponentialKernel(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_se_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_se_kernel.set_noise(self.noise)

        return copied_se_kernel

    @staticmethod
    @tf.function
    def get_l_derivative_matrix(hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> tf.Tensor:
        diff: tf.Tensor = aux_dist.manhattan_distance(x_vector, x_vector_)#tf.math.subtract(x_vector, tf.transpose(x_vector_))
        squared_distance: tf.Tensor = tf.square(diff, name="squared_distance")

        length_scale = hyper_parameter[0]
        l_squared: tf.Tensor = tf.square(length_scale, name="l_squared")
        cov_mat: tf.Tensor = \
            tf.math.exp(-0.5 * tf.math.divide(squared_distance, l_squared), name="SquaredExponentialKernel")

        return tf.multiply(tf.divide(diff, tf.multiply(length_scale, l_squared)), cov_mat)

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        return [self.get_l_derivative_matrix(hyper_parameter, x_vector, x_vector_)]

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        names = [representation + "_l"]

        if global_param.p_scaled_base_kernel:
            names.append(representation + "_sg")

        return names

    def type_compare_to(self, other):
        return isinstance(other, SquaredExponentialKernel)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(SquaredExponentialKernel, self).get_last_hyper_parameter()
        if scaling_x_param is None:
            return result

        hyper_param = [result[0] * scaling_x_param[1]]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(result[1])

        return hyper_param

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        super(SquaredExponentialKernel, self).set_last_hyper_parameter(last_hyper_parameter)

        self.last_hyper_parameter[0] = tf.math.abs(self.last_hyper_parameter[0])


class PeriodicKernel(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(PeriodicKernel, self).__init__(k.KernelManifestation.PER, input_dimensionality)
        self.latest_cov_mat: tf.Tensor = None

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self)

        with tf.name_scope("PeriodicKernel"):
            dist: tf.Tensor = aux_dist.manhattan_distance(x_vector, x_vector_)
            sine_input: tf.Tensor = tf.math.multiply(tf.cast(np.pi, global_param.p_dtype),
                                                     tf.math.divide(dist, hyper_parameter[1]))
            sine: tf.Tensor = tf.square(tf.math.sin(sine_input))
        result: tf.Tensor = tf.math.exp(-2 * sine / tf.math.square(hyper_parameter[0]), name="PeriodicKernel")

        if global_param.p_scaled_base_kernel:
            result = tf.multiply(hyper_parameter[2], result)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        range_length = xrange[0][1] - xrange[0][0]
        l_lower_bound = tf.cast((5 * range_length / n), global_param.p_dtype)
        l_upper_bound = tf.cast((range_length / 3), global_param.p_dtype)

        per_lower_bound = tf.cast(tf.math.log(10 * (range_length / n)), global_param.p_dtype)
        per_upper_bound = tf.cast(tf.math.log((range_length / 5)), global_param.p_dtype)

        if self.get_number_of_hyper_parameter() == 3:
            scale_lower_bound = tf.cast(global_param.p_cov_matrix_jitter * 100, dtype=global_param.p_dtype)
            scale_upper_bound = tf.constant(np.inf, dtype=global_param.p_dtype)
            return [(l_lower_bound, l_upper_bound), (per_lower_bound, per_upper_bound),
                    (scale_lower_bound, scale_upper_bound)]
        else:
            return [(l_lower_bound, l_upper_bound), (per_lower_bound, per_upper_bound)]

    def get_number_of_hyper_parameter(self) -> int:
        number_hyp = 2

        if global_param.p_scaled_base_kernel:
            number_hyp += 1

        return number_hyp

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        if from_distribution:
            return self.get_default_hyper_parameter_distribution(xrange, n)
        else:
            return self.get_default_hyper_parameter_fixed(xrange, n)

    def get_default_hyper_parameter_fixed(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        width = xrange[0][1] - xrange[0][0]
        initial_value = width / 10
        length_scale = cast(tf.Tensor, tf.Variable(initial_value, dtype=global_param.p_dtype, shape=hyp_dim[0]))
        p = cast(tf.Tensor, tf.Variable(initial_value, dtype=global_param.p_dtype, shape=hyp_dim[1]))
        hyper_param = [length_scale, p]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(tf.Variable(0.1, shape=hyp_dim[2], dtype=global_param.p_dtype))

        return hyper_param

    def get_default_hyper_parameter_distribution(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        default_hyp: List[tf.Tensor] = []

        width = xrange[0][1] - xrange[0][0]

        length_scale = tf.Variable(
            tf.abs(tf.random.normal(hyp_dim[0], mean=width / 10, stddev=0.2, dtype=global_param.p_dtype)))

        default_hyp.append(length_scale)

        t = tf.constant(xrange, dtype=global_param.p_dtype)
        avg_dist = tf.reduce_min(tf.gather(t, [1], axis=1) - tf.gather(t, [0], axis=1)) / n

        periodicity = tf.Variable(
            tf.random.uniform(hyp_dim[1], minval=avg_dist * 5, maxval=avg_dist * (n / 2), dtype=global_param.p_dtype))

        default_hyp.append(periodicity)

        if global_param.p_scaled_base_kernel:
            scale = tf.Variable(tf.abs(tf.random.normal(hyp_dim[2], mean=0.1, stddev=0.2, dtype=global_param.p_dtype)))
            default_hyp.append(scale)

        return default_hyp

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        width = xrange[0][1] - xrange[0][0]

        result.append({"shape": hyp_dim[0], "mean": width / 10, "stddev": 0.2, "type": "random_normal"})

        t = tf.constant(xrange, dtype=global_param.p_dtype)
        avg_dist = tf.reduce_min(tf.gather(t, [1], axis=1) - tf.gather(t, [0], axis=1)) / n

        result.append({"shape": hyp_dim[1], "minval": avg_dist * 5, "maxval": avg_dist * (n / 2),
                       "type": "random_uniform"})

        if global_param.p_scaled_base_kernel:
            result.append({"shape": hyp_dim[0], "mean": 0.1, "stddev": 0.2, "type": "random_normal"})

        return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        dimensionalities = [[], []]

        if global_param.p_scaled_base_kernel:
            dimensionalities.append([])

        return dimensionalities

    def deepcopy(self):
        copied_periodic_kernel = PeriodicKernel(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_periodic_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_periodic_kernel.set_noise(self.noise)

        return copied_periodic_kernel

    @staticmethod
    @tf.function
    def get_p_derivative_matrix(hyper_parameter, cov_mat, pi, dist, u, l_squared) -> tf.Tensor:
        two = tf.cast(2, dtype=global_param.p_dtype)
        sin = tf.sin(tf.multiply(two, u))

        frac_1 = tf.multiply(tf.multiply(two, pi), dist)

        frac_2 = tf.multiply(l_squared, tf.square(hyper_parameter[1]))

        frac = tf.divide(frac_1, frac_2)

        return tf.multiply(frac, tf.multiply(sin, cov_mat))

    @staticmethod
    @tf.function
    def get_l_derivative_matrix(hyper_parameter, cov_mat, exponent) -> tf.Tensor:
        frac = tf.multiply(tf.divide(-2, hyper_parameter[0]), exponent)

        return tf.multiply(frac, cov_mat)

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> \
            List[tf.Tensor]:
        pi = tf.cast(np.pi, global_param.p_dtype)
        dist = aux_dist.manhattan_distance(x_vector, x_vector_) #tf.math.subtract(tf.reshape(x_vector, [-1, 1]), tf.reshape(x_vector_, [1, -1]))
        u = tf.math.multiply(pi, tf.math.divide(dist, hyper_parameter[1]))
        sine = tf.square(tf.sin(u))
        l_squared = tf.math.square(hyper_parameter[0])
        exponent = tf.divide(tf.multiply(tf.cast(-2, dtype=global_param.p_dtype), sine), l_squared)
        cov_mat = tf.math.exp(exponent, name="PeriodicKernel")
        self.last_hyper_parameter = hyper_parameter
        self.latest_cov_mat = cov_mat
        return [self.get_l_derivative_matrix(hyper_parameter, cov_mat, exponent),
                self.get_p_derivative_matrix(hyper_parameter, cov_mat, pi, dist, u, l_squared)]

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        names = [representation + "_l", representation + "_p"]

        if global_param.p_scaled_base_kernel:
            names.append(representation + "_sg")

        return names

    def type_compare_to(self, other):
        return isinstance(other, PeriodicKernel)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(PeriodicKernel, self).get_last_hyper_parameter()
        if scaling_x_param is None:
            return result

        hyper_param = [result[0] * scaling_x_param[1], result[1] * scaling_x_param[1]]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(result[2])

        return hyper_param

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        super(PeriodicKernel, self).set_last_hyper_parameter(last_hyper_parameter)

        self.last_hyper_parameter[0] = tf.math.abs(self.last_hyper_parameter[0])

        self.last_hyper_parameter[1] = tf.math.abs(self.last_hyper_parameter[1])


class WhiteNoiseKernel(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(WhiteNoiseKernel, self).__init__(k.KernelManifestation.WN, input_dimensionality)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self)

        len_x: int = np.shape(x_vector)[0]
        len_x_: int = np.shape(x_vector_)[0]

        result = tf.zeros([len_x, len_x_], dtype=global_param.p_dtype)

        used_len_x: int

        if len_x > len_x_:
            used_len_x = len_x_
        else:
            used_len_x = len_x

        result = result + tf.eye(used_len_x, dtype=global_param.p_dtype)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_number_of_hyper_parameter(self) -> int:
        number_hyp = 0

        return number_hyp

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        return []

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        return []

    def deepcopy(self):
        copied_kernel = WhiteNoiseKernel(input_dimensionality=self.input_dimensionality)

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        return []

    def type_compare_to(self, other):
        return isinstance(other, WhiteNoiseKernel)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        return []

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        pass


class MaternKernel3_2(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(MaternKernel3_2, self).__init__(k.KernelManifestation.MAT32, input_dimensionality)
        self.latest_cov_mat: tf.Tensor = None

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), \
            "Invalid hyper_param size: " + str(self)

        with tf.name_scope("Matern_3_2_Kernel"):
            dist: tf.Tensor = aux_dist.manhattan_distance(x_vector, x_vector_)

            l = tf.abs(hyper_parameter[0])
            frac: tf.Tensor = tf.divide(tf.sqrt(tf.cast(3, dtype=global_param.p_dtype)) * dist, l)

            result: tf.Tensor = (tf.cast(1, dtype=global_param.p_dtype) + frac) * tf.exp(-frac)

        if global_param.p_scaled_base_kernel:
            result = tf.multiply(hyper_parameter[1], result)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        range_length = xrange[0][1] - xrange[0][0]
        l_lower_bound = tf.cast((5 * range_length / n), global_param.p_dtype)
        l_upper_bound = tf.cast((range_length / 3), global_param.p_dtype)

        if self.get_number_of_hyper_parameter() == 2:
            scale_lower_bound = tf.cast(global_param.p_cov_matrix_jitter * 100, dtype=global_param.p_dtype)
            scale_upper_bound = tf.constant(np.inf, dtype=global_param.p_dtype)
            return [(l_lower_bound, l_upper_bound), (scale_lower_bound, scale_upper_bound)]
        else:
            return [(l_lower_bound, l_upper_bound)]

    def get_number_of_hyper_parameter(self) -> int:
        number_hyp = 1

        if global_param.p_scaled_base_kernel:
            number_hyp += 1

        return number_hyp

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        if from_distribution:
            return self.get_default_hyper_parameter_distribution(xrange, n)
        else:
            return self.get_default_hyper_parameter_fixed(xrange, n)

    def get_default_hyper_parameter_fixed(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        width = xrange[0][1] - xrange[0][0]
        initial_value = width / 10
        length_scale = cast(tf.Tensor, tf.Variable(initial_value, dtype=global_param.p_dtype, shape=hyp_dim[0]))
        hyper_param = [length_scale]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(tf.Variable(0.1, shape=hyp_dim[1], dtype=global_param.p_dtype))

        return hyper_param

    def get_default_hyper_parameter_distribution(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        default_hyp: List[tf.Tensor] = []

        width = xrange[0][1] - xrange[0][0]

        length_scale = tf.Variable(
            tf.abs(tf.random.normal(hyp_dim[0], mean=width / 10, stddev=0.2, dtype=global_param.p_dtype)))

        default_hyp.append(length_scale)

        if global_param.p_scaled_base_kernel:
            scale = tf.Variable(tf.abs(tf.random.normal(hyp_dim[1], mean=0.1, stddev=0.2, dtype=global_param.p_dtype)))
            default_hyp.append(scale)

        return default_hyp

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        width = xrange[0][1] - xrange[0][0]

        result.append({"shape": hyp_dim[0], "mean": width / 10, "stddev": 0.2, "type": "random_normal"})

        if global_param.p_scaled_base_kernel:
            result.append({"shape": hyp_dim[0], "mean": 0.1, "stddev": 0.2, "type": "random_normal"})

        return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        dimensionalities = [[]]

        if global_param.p_scaled_base_kernel:
            dimensionalities.append([])

        return dimensionalities

    def deepcopy(self):
        copied_kernel = MaternKernel3_2(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_p_derivative_matrix(self, hyper_parameter, cov_mat, pi, dist, u, l_squared) -> tf.Tensor:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_l_derivative_matrix(self, hyper_parameter, cov_mat, exponent) -> tf.Tensor:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> \
            List[tf.Tensor]:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        names = [representation + "_l"]

        if global_param.p_scaled_base_kernel:
            names.append(representation + "_sg")

        return names

    def type_compare_to(self, other):
        return isinstance(other, MaternKernel3_2)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(MaternKernel3_2, self).get_last_hyper_parameter()
        if scaling_x_param is None:
            return result

        hyper_param = [result[0] * scaling_x_param[1]]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(result[1])

        return hyper_param

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        super(MaternKernel3_2, self).set_last_hyper_parameter(last_hyper_parameter)

        self.last_hyper_parameter[0] = tf.math.abs(self.last_hyper_parameter[0])


class MaternKernel5_2(BaseKernel):
    def __init__(self, input_dimensionality: int):
        super(MaternKernel5_2, self).__init__(k.KernelManifestation.MAT52, input_dimensionality)
        self.latest_cov_mat: tf.Tensor = None

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self)

        with tf.name_scope("Matern_5_2_Kernel"):
            dist: tf.Tensor = aux_dist.manhattan_distance(x_vector, x_vector_)

            l = tf.abs(hyper_parameter[0])
            frac: tf.Tensor = tf.divide(tf.sqrt(tf.cast(5, dtype=global_param.p_dtype)) * dist, l)

            third_summand: tf.Tensor = tf.divide(tf.cast(5, dtype=global_param.p_dtype) * tf.square(dist),
                                                 tf.cast(3, dtype=global_param.p_dtype) * tf.square(l))

            result: tf.Tensor = (tf.cast(1, dtype=global_param.p_dtype) + frac + third_summand) * tf.exp(-frac)

        if global_param.p_scaled_base_kernel:
            result = tf.multiply(hyper_parameter[1], result)

        self.last_hyper_parameter = hyper_parameter

        return result

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        range_length = xrange[0][1] - xrange[0][0]
        l_lower_bound = tf.cast((5 * range_length / n), global_param.p_dtype)
        l_upper_bound = tf.cast((range_length / 3), global_param.p_dtype)

        if self.get_number_of_hyper_parameter() == 2:
            scale_lower_bound = tf.cast(global_param.p_cov_matrix_jitter * 100, dtype=global_param.p_dtype)
            scale_upper_bound = tf.constant(np.inf, dtype=global_param.p_dtype)
            return [(l_lower_bound, l_upper_bound), (scale_lower_bound, scale_upper_bound)]
        else:
            return [(l_lower_bound, l_upper_bound)]

    def get_number_of_hyper_parameter(self) -> int:
        number_hyp = 1

        if global_param.p_scaled_base_kernel:
            number_hyp += 1

        return number_hyp

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        if from_distribution:
            return self.get_default_hyper_parameter_distribution(xrange, n)
        else:
            return self.get_default_hyper_parameter_fixed(xrange, n)

    def get_default_hyper_parameter_fixed(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()
        width = xrange[0][1] - xrange[0][0]
        initial_value = width / 10
        length_scale = cast(tf.Tensor, tf.Variable(initial_value, dtype=global_param.p_dtype, shape=hyp_dim[0]))
        hyper_param = [length_scale]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(tf.Variable(0.1, shape=hyp_dim[1], dtype=global_param.p_dtype))

        return hyper_param

    def get_default_hyper_parameter_distribution(self, xrange: List[List[float]], n: int) -> List[tf.Tensor]:
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        default_hyp: List[tf.Tensor] = []

        width = xrange[0][1] - xrange[0][0]

        length_scale = tf.Variable(
            tf.abs(tf.random.normal(hyp_dim[0], mean=width / 10, stddev=0.2, dtype=global_param.p_dtype)))

        default_hyp.append(length_scale)

        if global_param.p_scaled_base_kernel:
            scale = tf.Variable(tf.abs(tf.random.normal(hyp_dim[1], mean=0.1, stddev=0.2, dtype=global_param.p_dtype)))
            default_hyp.append(scale)

        return default_hyp

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []
        hyp_dim = self.get_hyper_parameter_dimensionalities()

        width = xrange[0][1] - xrange[0][0]

        result.append({"shape": hyp_dim[0], "mean": width / 10, "stddev": 0.2, "type": "random_normal"})

        if global_param.p_scaled_base_kernel:
            result.append({"shape": hyp_dim[0], "mean": 0.1, "stddev": 0.2, "type": "random_normal"})

        return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        dimensionalities = [[]]

        if global_param.p_scaled_base_kernel:
            dimensionalities.append([])

        return dimensionalities

    def deepcopy(self):
        copied_kernel = MaternKernel5_2(input_dimensionality=self.input_dimensionality)
        if self.last_hyper_parameter is not None:
            copied_kernel.set_last_hyper_parameter(self.last_hyper_parameter)

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_p_derivative_matrix(self, hyper_parameter, cov_mat, pi, dist, u, l_squared) -> tf.Tensor:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_l_derivative_matrix(self, hyper_parameter, cov_mat, exponent) -> tf.Tensor:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> \
            List[tf.Tensor]:
        raise Exception("get_p_derivative_matrix not implemented for Matern_3_2")

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        representation = self.get_string_representation()

        if kernel_id >= 0:
            representation += "_%i" % kernel_id

        names = [representation + "_l"]

        if global_param.p_scaled_base_kernel:
            names.append(representation + "_sg")

        return names

    def type_compare_to(self, other):
        return isinstance(other, MaternKernel5_2)

    def get_last_hyper_parameter(self, scaling_x_param=None):
        result = super(MaternKernel5_2, self).get_last_hyper_parameter()
        if scaling_x_param is None:
            return result

        hyper_param = [result[0] * scaling_x_param[1]]

        if global_param.p_scaled_base_kernel:
            hyper_param.append(result[1])

        return hyper_param

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        super(MaternKernel5_2, self).set_last_hyper_parameter(last_hyper_parameter)

        self.last_hyper_parameter[0] = tf.math.abs(self.last_hyper_parameter[0])