from enum import Enum
from typing import List

import numpy as np
import tensorflow as tf

import gpbasics.Auxiliary.BasicGPComponent as bgpc


class ConstantHyperParamType(Enum):
    NONE_CONSTANT = 0
    JUST_CONSTANT_BASE_KERNELS = 1
    JUST_CONSTANT_CP = 2
    ALL_CONSTANT = 3


class KernelType(Enum):
    BASE_KERNEL = 1
    OPERATOR = 2


class KernelManifestation(Enum):
    C = 101
    LIN = 102
    RQ = 103
    PER = 104
    SE = 105
    WN = 106
    MAT32 = 107
    MAT52 = 108

    ADD = 201
    MUL = 202
    CP = 203
    PART = 204


class Kernel(bgpc.Component):
    """
    Represents the general notion of a kernel used for a Gaussian Process.
    """

    def __init__(self, kernel_type: KernelType, manifestation: KernelManifestation, input_dimensionality: int):
        assert input_dimensionality >= 1, "input_dimensionality for a kernel ought to be one or larger"
        self.kernel_type: KernelType = kernel_type
        self.manifestation: KernelManifestation = manifestation
        self.last_hyper_parameter: List[tf.Tensor] = None
        self.input_dimensionality: int = input_dimensionality
        self.noise: tf.Tensor = None

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        pass

    def get_kernel_type(self) -> KernelType:
        return self.kernel_type

    def get_kernel_manifestation(self) -> KernelManifestation:
        return self.manifestation

    def get_number_of_hyper_parameter(self) -> int:
        pass

    def get_string_representation(self) -> str:
        pass

    def get_number_base_kernels(self) -> int:
        pass

    def get_default_hyper_parameter(
            self, xrange: List[List[float]], n: int, from_distribution: bool = False) -> List[tf.Tensor]:
        pass

    def set_last_hyper_parameter(self, last_hyper_parameter: List[tf.Tensor]):
        pass

    def get_last_hyper_parameter(self, scaling_x_param=None):
        pass

    def set_noise(self, noise: tf.Tensor):
        if noise.shape == []:
            self.noise = noise
        else:
            raise Exception("Invalid Noise set for Kernel")

    def get_noise(self):
        return self.noise

    def deepcopy(self):
        pass

    def get_string_representation_weight(self) -> float:
        return 0

    def sort_child_nodes(self):
        pass

    def get_json(self) -> dict:
        pass

    def get_number_of_child_nodes(self) -> int:
        pass

    def get_derivative_matrices(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) \
            -> List[tf.Tensor]:
        pass

    def get_hyper_parameter_names(self, kernel_id: int = -1) -> List[str]:
        pass

    def get_dimensionality(self):
        return self.input_dimensionality

    def set_dimensionality(self, input_dimensionality: int):
        self.input_dimensionality = input_dimensionality

    def get_simplified_version(self):
        return self

    def type_compare_to(self, other):
        return self == other

    def get_hash_tuple(self):
        hyp_as_plain_floats = []

        if self.last_hyper_parameter is not None and isinstance(self.last_hyper_parameter, list):
            for hyp in self.last_hyper_parameter:
                hyp = hyp.numpy().tolist()

                if isinstance(hyp, float):
                    hyp_as_plain_floats.append(hyp)
                elif isinstance(hyp, list):
                    hyp_as_plain_floats.extend(hyp)

        if self.noise is None:
            return self.manifestation.value, None, tuple(hyp_as_plain_floats)

        return self.manifestation.value, float(self.noise), tuple(hyp_as_plain_floats)

    def __hash__(self):
        return hash(self.get_hash_tuple())
