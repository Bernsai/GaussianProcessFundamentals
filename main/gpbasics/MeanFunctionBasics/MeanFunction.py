from enum import Enum
from typing import List

import numpy as np
import tensorflow as tf

from gpbasics.Auxiliary import BasicGPComponent as bgpc


class ConstantHyperParamType(Enum):
    NONE_CONSTANT = 0
    ALL_CONSTANT = 3


class MeanFunctionType(Enum):
    BASE_MEAN_FUNCTION = 1
    OPERATOR = 2


class MeanFunctionManifestation(Enum):
    C = 101
    LIN = 102
    EXP = 103
    LOGIT = 104

    ADD = 201
    MUL = 202
    CP = 203


class MeanFunction(bgpc.Component):
    """
    Represents the general notion of a MeanFunction used for a Gaussian Process.
    """

    def __init__(self, mean_function_type: MeanFunctionType, manifestation: MeanFunctionManifestation,
                 input_dimensionality: int):
        assert input_dimensionality >= 1, "input_dimensionality for a mean function ought to be 1 or larger"
        self.type: MeanFunctionType = mean_function_type
        self.manifestation: MeanFunctionManifestation = manifestation
        self.last_hyper_parameter: List[tf.Tensor] = None
        self.input_dimensionality = input_dimensionality

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray) -> tf.Tensor:
        pass

    def get_mean_function_type(self) -> MeanFunctionType:
        return self.type

    def get_mean_function_manifestation(self) -> MeanFunctionManifestation:
        return self.manifestation

    def get_number_of_hyper_parameter(self) -> int:
        pass

    def get_string_representation(self) -> str:
        pass

    def get_string_representation_weight(self) -> int:
        pass

    def get_number_base_mean_function(self) -> int:
        pass

    def set_last_hyper_parameter(self, last_hyper_parameter):
        self.last_hyper_parameter = last_hyper_parameter

    def get_last_hyper_parameter(self) -> List[tf.Tensor]:
        return self.last_hyper_parameter

    def deepcopy(self):
        pass

    def get_default_hyper_parameter(self) -> List[tf.Tensor]:
        pass
