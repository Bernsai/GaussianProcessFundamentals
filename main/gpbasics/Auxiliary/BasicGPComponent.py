from typing import List, Tuple

import tensorflow as tf


class Component:
    def get_hyper_parameter_bounds(self, xrange: List[List[float]], n: int) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        pass

    def get_hyper_parameter_dimensionalities(self) -> List[list]:
        pass

    def get_hyper_parameter_distribution_definition(self, xrange: List[List[float]],  n: int) -> List[dict]:
        pass

    @staticmethod
    def serialize_hyper_parameter(hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        processed_hyper_parameters: List[tf.Tensor] = []

        for h in hyper_parameter:
            processed_hyper_parameters.append(tf.reshape(h, shape=[-1, ]))

        return tf.concat(processed_hyper_parameters, axis=0)

    @staticmethod
    def deserialize_hyper_parameter(hyper_parameter: tf.Tensor, dimensionalities: List[list]) -> List[tf.Tensor]:
        index = 0

        list_hyper_parameters: List[tf.Tensor] = []

        for dim in dimensionalities:
            if len(dim) == 0:
                size = 1
            else:
                size = dim[0]

            h = tf.reshape(tf.slice(hyper_parameter, [0], [size]), shape=dim)
            list_hyper_parameters.append(h)

            index += size

        return list_hyper_parameters
