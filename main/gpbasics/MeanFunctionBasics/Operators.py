from typing import List

import numpy as np
import tensorflow as tf

import gpbasics.MeanFunctionBasics.MeanFunction as mf

class Operator(mf.MeanFunction):
    def __init__(self, manifestation: mf.MeanFunctionManifestation, child_nodes: List[mf.MeanFunction],
                 input_dimensionality: int, sign: str):
        super(Operator, self).__init__(mf.MeanFunctionType.OPERATOR, manifestation, input_dimensionality)

        self.child_nodes: List[mf.MeanFunction] = child_nodes

        if self.manifestation.value < 200:
            print("Invalid manifestation for Operator: ", manifestation)

        self.sortable = False
        self.sign: str = sign

    def get_number_of_hyper_parameter(self):
        result = 0
        for cn in self.child_nodes:
            result = result + cn.get_number_of_hyper_parameter()
        return result

    def add_child_mean_function(self, mean_function):
        self.child_nodes = self.child_nodes + [mean_function]

    def replace_child_node(self, index, new_child_node: mf.MeanFunction):
        assert index < len(self.child_nodes), \
            "cannot replace child node at index" + str(index) + ". Invalid as there are only " + \
            str(len(self.child_nodes)) + " child nodes."
        self.child_nodes[index] = new_child_node

    def get_number_base_mean_function(self):
        number_base_mean_functions = 0
        for cn in self.child_nodes:
            number_base_mean_functions += cn.get_number_base_mean_function()

        return number_base_mean_functions

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        result = []

        for cn in self.child_nodes:
            result = result + cn.get_default_hyper_parameter()

        return result

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        assert len(last_hyper_parameter) == self.get_number_of_hyper_parameter(), \
            "Invalid hyper_param size: " + str(self)
        if len(self.child_nodes) == 1:
            return self.child_nodes[0].set_last_hyper_parameter(last_hyper_parameter)
        else:
            index = 0
            slice_hyper_param = last_hyper_parameter[
                                index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]
            self.child_nodes[0].set_last_hyper_parameter(slice_hyper_param)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param = last_hyper_parameter[
                                    index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                self.child_nodes[i].set_last_hyper_parameter(slice_hyper_param)
                index += self.child_nodes[i].get_number_of_hyper_parameter()

    def get_last_hyper_parameter(self) -> List[tf.Tensor]:
        result = []

        for cn in self.child_nodes:
            result = result + cn.get_last_hyper_parameter()

        return result

    def get_number_of_child_nodes(self):
        return len(self.child_nodes)

    def sort_child_nodes(self):
        if self.sortable:
            self.child_nodes = \
                sorted(self.child_nodes, key=lambda node: node.get_string_representation_weight(), reverse=False)

        for cn in self.child_nodes:
            if isinstance(cn, Operator):
                cn.sort_child_nodes()

    def get_string_representation_weight(self):
        weight = 0
        for cn in self.child_nodes:
            weight += cn.get_string_representation_weight()

        return weight

    def get_string_representation(self):
        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_string_representation()
        else:
            result = "("
            for i in range(0, len(self.child_nodes)):
                result = result + self.child_nodes[i].get_string_representation()
                if i < (len(self.child_nodes) - 1):
                    result = result + self.sign

            result = result + ")"
            return result

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        result = []

        for cn in self.child_nodes:
            result.extend(cn.get_hyper_parameter_dimensionalities())

        return result


class MultiplicationOperator(Operator):
    def __init__(self, child_nodes: List[mf.MeanFunction], input_dimensionality: int):
        super(MultiplicationOperator, self).__init__(
            mf.MeanFunctionManifestation.MUL, child_nodes, input_dimensionality, " x ")
        self.sortable = True

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray):
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(self)

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_tf_tensor(hyper_parameter, x_vector)
        else:
            index = 0
            slice_hyper_param = hyper_parameter[index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]
            result = self.child_nodes[0].get_tf_tensor(slice_hyper_param, x_vector)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param = hyper_parameter[index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                result = tf.math.multiply(result, self.child_nodes[i].get_tf_tensor(slice_hyper_param, x_vector),
                                          name="MUL_OP")
                index += self.child_nodes[i].get_number_of_hyper_parameter()

        return result

    def deepcopy(self):
        deep_copied_child_nodes = []

        for cn in self.child_nodes:
            deep_copied_child_nodes.append(cn.deepcopy())

        return MultiplicationOperator(deep_copied_child_nodes, self.input_dimensionality)


class AdditionOperator(Operator):
    def __init__(self, child_nodes: List[mf.MeanFunction], input_dimensionality: int):
        super(AdditionOperator, self).__init__(
            mf.MeanFunctionManifestation.ADD, child_nodes, input_dimensionality, " + ")
        self.sortable = True

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray):
        assert x_vector is not None, "Input vector x uninitialized: " + str(self)
        assert len(hyper_parameter) == self.get_number_of_hyper_parameter(), "Invalid hyper_param size: " + str(
            self.get_string_representation()) + str(len(hyper_parameter)) + "!=" + str(
            self.get_number_of_hyper_parameter())

        if len(self.child_nodes) == 1:
            return self.child_nodes[0].get_tf_tensor(hyper_parameter, x_vector)
        else:
            index = 0
            slice_hyper_param = hyper_parameter[index:(index + self.child_nodes[0].get_number_of_hyper_parameter())]
            result = self.child_nodes[0].get_tf_tensor(slice_hyper_param, x_vector)
            index += self.child_nodes[0].get_number_of_hyper_parameter()
            for i in range(1, len(self.child_nodes)):
                slice_hyper_param = hyper_parameter[index:(index + self.child_nodes[i].get_number_of_hyper_parameter())]
                result = tf.math.add(result, self.child_nodes[i].get_tf_tensor(slice_hyper_param, x_vector),
                                     name="ADD_OP")
                index += self.child_nodes[i].get_number_of_hyper_parameter()

        return result

    def deepcopy(self):
        deep_copied_child_nodes = []

        for cn in self.child_nodes:
            deep_copied_child_nodes.append(cn.deepcopy())

        return AdditionOperator(deep_copied_child_nodes, self.input_dimensionality)
