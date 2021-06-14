from typing import List

import numpy as np

from gpbasics.DataHandling.Normalization import NormalizationStructure as norm


class MinMaxNormalization(norm.Normalization):
    def __init__(self, min_x: List[float], max_x: List[float], min_y: float, max_y: float):
        assert len(min_x) == len(max_x)
        self.min_x: np.ndarray = np.array(min_x)
        self.max_x: np.ndarray = np.array(max_x)
        self.min_y: float = min_y
        self.max_y: float = max_y

    def normalize_x(self, x_vector: np.ndarray):
        assert x_vector.shape[1] == self.min_x.shape[0]

        normalized_x_vector = (x_vector - self.min_x) / self.max_x

        return normalized_x_vector

    def denormalize_x(self, x_vector: np.ndarray):
        assert x_vector.shape[1] == self.min_x.shape[0]

        denormalized_x_vector = (x_vector * self.max_x) + self.min_x

        return denormalized_x_vector

    def normalize_y(self, y_vector: np.ndarray):
        normalized_y_vector = (y_vector - self.min_y) / self.max_y

        return normalized_y_vector

    def denormalize_y(self, y_vector: np.ndarray):
        denormalized_y_vector = (y_vector * self.max_y) + self.min_y

        return denormalized_y_vector
