import logging
from functools import reduce
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import gpbasics.KernelBasics.Kernel as k
import gpbasics.KernelBasics.BaseKernels as bk
import gpbasics.global_parameters as global_param

global_param.ensure_init()


class Operator(k.Kernel):
    def __init__(self, manifestation: k.KernelManifestation, input_dimensionality: int, child_nodes: List[k.Kernel]):
        super(Operator, self).__init__(k.KernelType.OPERATOR, manifestation, input_dimensionality)

        self.child_nodes: List[k.Kernel] = child_nodes

        self.operator_sign: str = "UNKNOWN"

        if self.manifestation.value < 200:
            logging.critical("Invalid manifestation for Operator: ", manifestation)

        self.sortable: bool = False

    def get_number_of_hyper_parameter(self) -> int:
        result: int = 0
        for cn in self.child_nodes:
            result = result + cn.get_number_of_hyper_parameter()
        return result

    def add_kernel(self, kernel):
        self.child_nodes = self.child_nodes + [kernel]

    def replace_child_node(self, index: int, new_child_node: k.Kernel):
        assert index < len(self.child_nodes), \
            "cannot replace child node at index" + str(index) + ". Invalid as there are only " + \
            str(len(self.child_nodes)) + " child nodes."
        self.child_nodes[index] = new_child_node

    def get_number_base_kernels(self) -> int:
        number_base_kernels: int = 0
        for cn in self.child_nodes:
            number_base_kernels += cn.get_number_base_kernels()
        return number_base_kernels

    def get_string_representation(self) -> str:
        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_string_representation()
        else:
            result: str = "("
            for i in range(0, len(self.child_nodes)):
                result = result + self.child_nodes[i].get_string_representation()
                if i < (len(self.child_nodes) - 1):
                    result = result + " " + self.operator_sign + " "

            result = result + ")"
            return result

    def get_string_representation_weight(self) -> int:
        weight = 0
        for cn in self.child_nodes:
            weight += cn.get_string_representation_weight()

        return weight

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        result = []

        for cn in self.child_nodes:
            result = result + cn.get_default_hyper_parameter(xrange, n, from_distribution)

        return result

    def get_json(self) -> dict:
        if len(self.child_nodes) == 1:
            return {"type": self.manifestation.name, "child_nodes": [self.child_nodes[0].get_json()]}

        json_child_nodes: list = []
        for cn in self.child_nodes:
            json_child_nodes.append(cn.get_json())

        return {"type": self.manifestation.name, "child_nodes": json_child_nodes}

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        assert len(last_hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self)
        if len(self.child_nodes) == 1:
            return self.child_nodes[0].set_last_hyper_parameter(last_hyper_parameter)
        else:
            index = 0

            slice_hyper_param: List[tf.Tensor] = \
                last_hyper_parameter[index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]

            self.child_nodes[0].set_last_hyper_parameter(slice_hyper_param)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param: List[tf.Tensor] = \
                    last_hyper_parameter[index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                self.child_nodes[i].set_last_hyper_parameter(slice_hyper_param)
                index += self.child_nodes[i].get_number_of_hyper_parameter()

    def get_last_hyper_parameter(self, scaling_x_param=None):
        last_hyper_parameter = []
        for cn in self.child_nodes:
            hyper_parameter = cn.get_last_hyper_parameter(scaling_x_param)
            last_hyper_parameter.extend(hyper_parameter)

        return last_hyper_parameter

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        hyperparameter_bounds = []

        for cn in self.child_nodes:
            hyperparameter_bounds.extend(cn.get_hyper_parameter_bounds(xrange, n))

        return hyperparameter_bounds

    def set_noise(self, noise: tf.Tensor):
        super(Operator, self).set_noise(noise)
        for cn in self.child_nodes:
            cn.set_noise(noise)

    def sort_child_nodes(self):
        if self.sortable:
            self.child_nodes = \
                sorted(self.child_nodes, key=lambda node: node.get_string_representation_weight(), reverse=False)

        for cn in self.child_nodes:
            if isinstance(cn, Operator):
                cn.sort_child_nodes()

    def get_number_of_child_nodes(self) -> int:
        return len(self.child_nodes)

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        hyper_parameter_names: List[str] = []

        for cn in self.child_nodes:
            hyper_parameter_names += cn.get_hyper_parameter_names(kernel_id)
            if isinstance(cn, bk.BaseKernel) and kernel_id >= 0:
                kernel_id += 1

        return hyper_parameter_names

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        result = []

        for cn in self.child_nodes:
            result.extend(cn.get_hyper_parameter_dimensionalities())

        return result

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        result = []

        for cn in self.child_nodes:
            result.extend(cn.get_hyper_parameter_distribution_definition(xrange, n))

        return result

    def set_dimensionality(self, input_dimensionality: int):
        super(Operator, self).set_dimensionality(input_dimensionality)

        for cn in self.child_nodes:
            cn.set_dimensionality(input_dimensionality)

    def get_simplified_version(self):
        pass

    def type_compare_to(self, other):
        while isinstance(other, Operator) and len(other.child_nodes) == 1:
            other = other.child_nodes[0]

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].type_compare_to(other)

        if not isinstance(other, Operator):
            return False

        if len(other.child_nodes) != len(self.child_nodes):
            return False

        self.sort_child_nodes()
        other.sort_child_nodes()

        for idx, cn in enumerate(self.child_nodes):
            if not cn.type_compare_to(other.child_nodes[idx]):
                return False

        return True

    def get_hash_tuple(self):
        return super(Operator, self).get_hash_tuple() + (sum([hash(cn) for cn in self.child_nodes]), )


class MultiplicationOperator(Operator):
    def __init__(self, input_dimensionality: int, child_nodes: List[k.Kernel]):
        super(MultiplicationOperator, self).__init__(k.KernelManifestation.MUL, input_dimensionality, child_nodes)
        self.operator_sign = "x"
        self.sortable = True

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: %s" % str(self)

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_tf_tensor(hyper_parameter, x_vector, x_vector_)
        else:
            index: int = 0
            slice_hyper_param: List[tf.Tensor] = \
                hyper_parameter[index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]
            result: tf.Tensor = self.child_nodes[0].get_tf_tensor(slice_hyper_param, x_vector, x_vector_)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param = hyper_parameter[index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                result = tf.math.multiply(result, self.child_nodes[i].get_tf_tensor(
                    slice_hyper_param, x_vector, x_vector_), name="MUL_OP")
                index += self.child_nodes[i].get_number_of_hyper_parameter()

        return result

    def deepcopy(self):
        deep_copied_child_nodes = []

        for cn in self.child_nodes:
            deep_copied_child_nodes.append(cn.deepcopy())

        copied_kernel = MultiplicationOperator(self.input_dimensionality, deep_copied_child_nodes)

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Got wrong number of hyper params in MUL"
        ks: List[tf.Tensor] = []
        ks_dev: List[tf.Tensor] = []
        derivative_matrices: List[tf.Tensor] = []
        index: int = 0
        for i, cn in enumerate(self.child_nodes):
            hyper_param_count = cn.get_number_of_hyper_parameter()
            slice_hyper_param = hyper_parameter[index:(index + hyper_param_count)]
            ks.append(cn.get_tf_tensor(slice_hyper_param, x_vector, x_vector_))
            ks_dev.extend(cn.get_derivative_matrices(slice_hyper_param, x_vector, x_vector_))
            index += hyper_param_count

        child_index: int = 0
        parent_index: int = 0
        for i in range(len(hyper_parameter)):
            k_factors: List[tf.Tensor] = ks[:parent_index] + ks[parent_index + 1:]
            k_prime: tf.Tensor = ks_dev[i]
            k_factors.append(k_prime)
            child_index += 1
            if child_index >= self.child_nodes[parent_index].get_number_of_hyper_parameter():
                child_index = 0
                parent_index += 1

            dev_mat: tf.Tensor = reduce(lambda prod, fac: tf.multiply(prod, fac), k_factors)

            derivative_matrices.append(dev_mat)

        return derivative_matrices

    def get_simplified_version(self):
        child_nodes = [cn.get_simplified_version() for cn in self.child_nodes]

        simplified_child_nodes = []

        addition_indices = []

        for cn in child_nodes:
            if isinstance(cn, MultiplicationOperator):
                simplified_child_nodes.extend(cn.child_nodes)
            else:
                if isinstance(cn, AdditionOperator):
                    addition_indices.append(len(simplified_child_nodes))
                simplified_child_nodes.append(cn)

        if len(addition_indices) == 0:
            return MultiplicationOperator(self.input_dimensionality, simplified_child_nodes)

        addition_child_nodes = []

        mul_child_nodes = [simplified_child_nodes[i] for i in range(len(simplified_child_nodes)) if
                           i != addition_indices[0]]

        for mcn in simplified_child_nodes[addition_indices[0]].child_nodes:
            addition_child_nodes.append(MultiplicationOperator(self.input_dimensionality, mul_child_nodes + [mcn]))

        return AdditionOperator(self.input_dimensionality, addition_child_nodes).get_simplified_version()


class AdditionOperator(Operator):
    def __init__(self, input_dimensionality: int, child_nodes: List[k.Kernel]):
        super(AdditionOperator, self).__init__(k.KernelManifestation.ADD, input_dimensionality, child_nodes)
        self.operator_sign = "+"
        self.sortable = True

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self)

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_tf_tensor(hyper_parameter, x_vector, x_vector_)
        else:
            index = 0
            slice_hyper_param: List[tf.Tensor] = \
                hyper_parameter[index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]
            result: tf.Tensor = self.child_nodes[0].get_tf_tensor(slice_hyper_param, x_vector, x_vector_)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param: List[tf.Tensor] = \
                    hyper_parameter[index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                result = tf.math.add(result, self.child_nodes[i].get_tf_tensor(
                    slice_hyper_param, x_vector, x_vector_), name="ADD_OP")
                index += self.child_nodes[i].get_number_of_hyper_parameter()

        return result

    def deepcopy(self):
        deep_copied_child_nodes = []

        for cn in self.child_nodes:
            deep_copied_child_nodes.append(cn.deepcopy())

        copied_kernel = AdditionOperator(self.input_dimensionality, deep_copied_child_nodes)

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        derivative_matrices: List[tf.Tensor] = []
        index: int = 0
        for i, cn in enumerate(self.child_nodes):
            hyper_param_count: int = cn.get_number_of_hyper_parameter()
            slice_hyper_param: List[tf.Tensor] = \
                hyper_parameter[index: index + hyper_param_count]
            derivative_matrices.extend(cn.get_derivative_matrices(slice_hyper_param, x_vector, x_vector_))
            index += hyper_param_count

        return derivative_matrices

    def get_simplified_version(self):
        child_nodes = [cn.get_simplified_version() for cn in self.child_nodes]

        simplified_child_nodes = []

        for cn in child_nodes:
            if isinstance(cn, AdditionOperator):
                simplified_child_nodes.extend(cn.child_nodes)
            else:
                simplified_child_nodes.append(cn)

        return AdditionOperator(self.input_dimensionality, simplified_child_nodes)


class ChangePointOperator(Operator):
    def __init__(self, input_dimensionality: int, child_nodes: List[k.Kernel], change_point_positions: List[tf.Tensor]):
        super(ChangePointOperator, self).__init__(k.KernelManifestation.CP, input_dimensionality, child_nodes)
        assert (len(child_nodes) - 1) == len(change_point_positions), \
            "Error. Change Point positions and/or their positions wrongly initialized."
        self.operator_sign = "]["

        self.change_point_positions: List[tf.Tensor] = change_point_positions

    @staticmethod
    def approx_indicator(x, cp) -> tf.Tensor:
        sigmoid_slope = tf.constant(100, dtype=global_param.p_dtype)  # Slope of sigmoid
        return tf.math.divide(tf.cast(1, dtype=global_param.p_dtype),
                              tf.math.add(tf.cast(1, dtype=global_param.p_dtype),
                                          tf.math.exp(
                                              tf.cast(-1, dtype=global_param.p_dtype) * sigmoid_slope * (x - cp))))

    @staticmethod
    def sigmoid(x, cp) -> tf.Tensor:
        sigmoid_slope = tf.constant(0.5, dtype=global_param.p_dtype)  # Slope of sigmoid
        s = tf.constant(0.0025, dtype=global_param.p_dtype)

        result = sigmoid_slope * (1 + tf.math.tanh(tf.math.divide(cp - x, s)))

        return result

    @staticmethod
    def indicator_function_tf_less(x, cp) -> tf.Tensor:
        result_bool = tf.reshape(tf.math.less(tf.reshape(x, [-1, ]), tf.fill([x.shape[0]], tf.reshape(cp, []))),
                                 [-1, 1])
        return tf.cast(result_bool, dtype=tf.float64)

    @staticmethod
    def indicator_function_relu_sign(x, cp) -> tf.Tensor:
        return tf.nn.relu(tf.math.sign(tf.math.subtract(x, cp)))

    # @tf.custom_gradient
    def indicator_function(self, x, cp) -> tf.Tensor:
        return self.indicator_function_tf_less(x, cp)  # , tf.gradients(self.sigmoid(x, cp), x)

    def get_cp_encapsulated_kernel(self, kernel, x_vector: np.ndarray, x_vector_: np.ndarray,
                                   hyper_param, previous_sigmoid, cp) -> Tuple[tf.Tensor, tf.Tensor]:
        # assert previous_sigmoid != cp, "Error in CP operator: sigmoid and _sigmoid cannot both be left out"

        kernel = kernel.get_tf_tensor(hyper_param, x_vector, x_vector_)
        with tf.name_scope("CP_OP"):
            # if previous_sigmoid != 0:
            kernel = tf.math.multiply(kernel, previous_sigmoid)

            if cp is not None:
                if global_param.p_cp_operator_type == global_param.ChangePointOperatorType.SIGMOID:
                    _x_ind: tf.Tensor = self.sigmoid(x_vector, cp)
                    _x_s_ind: tf.Tensor = self.sigmoid(x_vector_, cp)

                elif global_param.p_cp_operator_type == global_param.ChangePointOperatorType.APPROX_INDICATOR:
                    _x_ind: tf.Tensor = self.approx_indicator(x_vector, cp)
                    _x_s_ind: tf.Tensor = self.approx_indicator(x_vector_, cp)
                else:
                    _x_ind: tf.Tensor = self.indicator_function(x_vector, cp)
                    _x_s_ind: tf.Tensor = self.indicator_function(x_vector_, cp)

                ind: tf.Tensor = tf.matmul(_x_ind, tf.linalg.matrix_transpose(_x_s_ind))
                one = tf.cast(1, dtype=global_param.p_dtype)
                _ind: tf.Tensor = tf.matmul((tf.subtract(one, _x_ind)),
                                            tf.linalg.matrix_transpose((tf.subtract(one, _x_s_ind))))

                kernel: tf.Tensor = tf.math.multiply(kernel, ind)

                return kernel, _ind

        return kernel, tf.constant(0, dtype=global_param.p_dtype)

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        assert x_vector is not None and x_vector_ is not None, "Input vectors x and x_ uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: %s" % str(self)

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_tf_tensor(hyper_parameter, x_vector, x_vector_)
        else:
            previous_sigmoid: tf.Tensor

            change_point_positions: List[tf.Tensor] = hyper_parameter[0: len(self.change_point_positions)]

            index: int = len(self.change_point_positions)

            previous_sigmoid = tf.constant(1, dtype=global_param.p_dtype)

            result: List[tf.Tensor] = []

            for i in range(0, len(self.child_nodes)):
                cp = None
                if i < len(change_point_positions):
                    cp = change_point_positions[i]

                slice_hyper_param: List[tf.Tensor] = \
                    hyper_parameter[index: index + self.child_nodes[i].get_number_of_hyper_parameter()]

                part_cov_mat: tf.Tensor
                part_cov_mat, previous_sigmoid = \
                    self.get_cp_encapsulated_kernel(self.child_nodes[i], x_vector, x_vector_,
                                                    slice_hyper_param, previous_sigmoid, cp)

                index += self.child_nodes[i].get_number_of_hyper_parameter()

                result.append(part_cov_mat)

            return tf.math.add_n(result)

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        result: List[list] = super(ChangePointOperator, self).get_hyper_parameter_dimensionalities()
        cp_dims: List[list] = [[len(self.change_point_positions), ]]
        result = cp_dims + result
        return result

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        assert len(last_hyper_parameter) == self.get_number_of_hyper_parameter(), \
            "Invalid hyper_param size: %s" % str(last_hyper_parameter)

        if len(self.child_nodes) == 1:
            self.child_nodes[0].set_last_hyper_parameter(last_hyper_parameter)
        else:
            self.change_point_positions = last_hyper_parameter[0: len(self.change_point_positions)]
            self.last_hyper_parameter = self.change_point_positions

            index: int = len(self.change_point_positions)

            slice_hyper_param: List[tf.Tensor] = \
                last_hyper_parameter[index: index + self.child_nodes[0].get_number_of_hyper_parameter()]

            self.child_nodes[0].set_last_hyper_parameter(slice_hyper_param)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param: List[tf.Tensor] = \
                    last_hyper_parameter[index: index + self.child_nodes[i].get_number_of_hyper_parameter()]
                self.child_nodes[i].set_last_hyper_parameter(slice_hyper_param)
                index += self.child_nodes[i].get_number_of_hyper_parameter()

    def get_number_of_hyper_parameter(self) -> int:
        result: int = 0
        for cn in self.child_nodes:
            result = result + cn.get_number_of_hyper_parameter()
        return result + len(self.change_point_positions)

    def add_kernel(self, kernel: k.Kernel, new_cp_position: tf.Tensor):
        assert kernel is not None, "Adding None as kernel to ChangePoint is not allowed."
        assert new_cp_position is not None

        assert (len(self.change_point_positions) == 0) or \
               (tf.reduce_min(new_cp_position - self.change_point_positions[len(self.change_point_positions) - 1])
                > 0), "New Changepoints _must_ be larger in value than the former largest change point. " \
                      "Change points need to be strict in order. New CP: " + str(new_cp_position) + \
                      ", other CP: " + str(self.change_point_positions)
        self.child_nodes.append(kernel)
        self.change_point_positions.append(new_cp_position)

    # Every Expression (S) becomes (S' | S) by adding new kernel expression S'
    def add_preceding_kernel(self, kernel, new_cp_position: tf.Tensor):
        assert kernel is not None, "Adding None as kernel to ChangePoint is not allowed."
        assert new_cp_position is not None

        assert (len(self.change_point_positions) == 0) or \
               (tf.reduce_min(self.change_point_positions[len(self.change_point_positions) - 1] - new_cp_position)
                > 0), "New Changepoints _must_ be larger in value than the former largest change point. " \
                      "Change points need to be strict in order. New CP: " + str(new_cp_position) + \
                      ", other CP: " + str(self.change_point_positions)
        self.child_nodes = [kernel] + self.child_nodes
        self.change_point_positions = [new_cp_position] + self.change_point_positions

    def get_simplified_kernel(self, data_range: List[float]) -> Tuple[k.Kernel, bool]:
        # delete those changepoints and kernels coming after the first change_point that is out of data range
        # delete those changepoints and corresponding kernels, for whom the following change_points value is smaller
        # => this would mean that via change point adaptation one child_node was virtually deleted!
        to_be_deleted_change_points = []
        to_be_deleted_child_nodes = []

        change_points = self.change_point_positions

        if global_param.p_cp_operator_type == global_param.ChangePointOperatorType.INDICATOR or \
                global_param.p_cp_operator_type == global_param.ChangePointOperatorType.APPROX_INDICATOR:
            blurring = 0
        else:
            blurring = 4

        for i in range(0, len(change_points)):
            # change point i beyond maximum range
            if change_points[i] >= (data_range[1] + blurring):
                to_be_deleted_change_points.append(i)
                to_be_deleted_child_nodes.append(i + 1)

            # change point i beyond minimum range
            if change_points[i] <= (data_range[0] - blurring):
                to_be_deleted_change_points.append(i)
                to_be_deleted_child_nodes.append(i)

            # change point i "overtook" change point i + 1
            if (len(change_points) - 1 > i) and change_points[i] >= change_points[i + 1]:
                to_be_deleted_change_points.append(i)
                to_be_deleted_child_nodes.append(i + 1)

        new_child_nodes = []
        new_change_points = []

        if len(to_be_deleted_change_points) == 0:
            return self, False

        for i in range(0, len(change_points)):
            if i not in to_be_deleted_change_points:
                new_change_points.append(change_points[i].__copy__())

        for i in range(0, len(self.child_nodes)):
            if i not in to_be_deleted_child_nodes:
                new_child_nodes.append(self.child_nodes[i])

        assert (len(new_change_points) + 1) == len(new_child_nodes), \
            "Error in get_simplified_kernel, new_change_points: %s, new_child_nodes: %s" \
            % (str(new_change_points), str(new_child_nodes))

        return ChangePointOperator(self.input_dimensionality, new_child_nodes, new_change_points), True

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        result = []

        for cn in self.child_nodes:
            result = result + cn.get_default_hyper_parameter(xrange, n, from_distribution)

        return self.change_point_positions + result

    def deepcopy(self):
        deep_copied_child_nodes = []

        for cn in self.child_nodes:
            deep_copied_child_nodes.append(cn.deepcopy())

        if isinstance(self.change_point_positions, tf.Tensor):
            copied_cp_positions = self.change_point_positions.numpy().tolist().copy()
        else:
            copied_cp_positions = self.change_point_positions.copy()

        copied_change_point_operator = ChangePointOperator(
            self.input_dimensionality, deep_copied_child_nodes, copied_cp_positions)

        if self.noise is not None:
            copied_change_point_operator.set_noise(self.noise)

        return copied_change_point_operator

    def get_last_hyper_parameter(self, scaling_x_param=None) -> List[tf.Tensor]:
        last_hyper_parameter = []
        for cn in self.child_nodes:
            last_hyper_parameter.extend(cn.get_last_hyper_parameter(scaling_x_param))

        # if self.last_hyper_parameter is None:
        if isinstance(self.change_point_positions, tf.Tensor):
            cpp = [self.change_point_positions]
        else:
            cpp = self.change_point_positions

        return cpp + last_hyper_parameter

    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        hyperparameter_bounds = []

        range_length = xrange[0][1] - xrange[0][0]
        cp_lower_bound = tf.cast(xrange[0][0] - (1.5 * range_length), dtype=global_param.p_dtype)
        cp_upper_bound = tf.cast(xrange[0][1] + (1.5 * range_length), dtype=global_param.p_dtype)

        cp_bound_tuples = [(cp_lower_bound, cp_upper_bound)] * len(self.change_point_positions)

        for cn in self.child_nodes:
            hyperparameter_bounds.extend(cn.get_hyper_parameter_bounds(xrange, n))

        return cp_bound_tuples + hyperparameter_bounds

    def get_json(self) -> dict:
        if len(self.child_nodes) == 1:
            return {"type": self.manifestation.name, "child_nodes": [self.child_nodes[0].get_json()]}

        child_nodes = []
        for i in range(len(self.child_nodes)):
            cn_json = self.child_nodes[i].get_json()

            if i == 0:
                assert isinstance(self.change_point_positions[i], tf.Tensor) or \
                       isinstance(self.change_point_positions[i], tf.Variable)
                cn_json['start_index'] = 0
                cn_json['stop_index'] = self.change_point_positions[i].numpy().tolist()

            elif i == len(self.change_point_positions):
                cn_json['start_index'] = self.change_point_positions[i - 1].numpy().tolist()

                cn_json['stop_index'] = 1.0
            else:
                assert isinstance(self.change_point_positions[i], tf.Tensor) or \
                       isinstance(self.change_point_positions[i], tf.Variable)
                cn_json['start_index'] = self.change_point_positions[i - 1].numpy().tolist()

                cn_json['stop_index'] = self.change_point_positions[i].numpy().tolist()

            child_nodes.append(cn_json)

        result = {"type": self.manifestation.name, "child_nodes": child_nodes}

        return result

    def get_simplified_version(self):
        simplified_child_nodes = [cn.get_simplified_version() for cn in self.child_nodes]

        return ChangePointOperator(self.input_dimensionality, simplified_child_nodes, self.change_point_positions)

    def get_hash_tuple(self):
        return super(ChangePointOperator, self).get_hash_tuple() + tuple([hash(cn) for cn in self.child_nodes])
