import logging
from typing import List

import numpy as np
import tensorflow as tf

import gpbasics.global_parameters as global_param

global_param.ensure_init()

import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.MeanFunctionBasics.BaseMeanFunctions as bmf
import gpbasics.MeanFunctionBasics.MeanFunction as mf
from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput


def is_equidistant(input_vector: np.ndarray):
    diff = input_vector[:len(input_vector) - 1] - input_vector[1:]
    mean = np.mean(diff)
    diff_max = np.abs(np.max(diff) - mean)
    diff_min = np.abs(np.min(diff) - mean)
    allowed_error = 1 / (100 * len(input_vector))
    return diff_max < allowed_error and diff_min < allowed_error


class DataInput(AbstractDataInput):
    """
    DataInput encapsulates a given dataset and allows all required operations _after_ reading it from any source.
    """

    def __init__(self, data_x_train: np.ndarray, data_y_train: np.ndarray, data_x_test: np.ndarray = None,
                 data_y_test: np.ndarray = None, test_ratio: float = -1, seed: int = 3061941):
        if data_x_test is not None and data_y_test is not None:
            data_x_test = tf.constant(data_x_test, dtype=global_param.p_dtype)
            data_y_test = tf.constant(data_y_test, dtype=global_param.p_dtype)
        super(DataInput, self).__init__(
            tf.constant(data_x_train, dtype=global_param.p_dtype),
            tf.constant(data_y_train, dtype=global_param.p_dtype), data_x_test,
            data_y_test, test_ratio, seed)

    def get_inducting_x_train(self, indices) -> tf.Tensor:
        """
        This function retrieves a uniform random sample of the given input (x) training data points.
        Underlying probability for picking each data point is predefined by global_param.p_nystroem_ratio.
        :return: tf.Tensor of shape [self.n_inducting_train, 1]
        """

        self.inducting_x_train = tf.gather(self.data_x_train, indices)

        return self.inducting_x_train

    def get_inducting_x_test(self, indices) -> tf.Tensor:
        """
        This function retrieves a uniform random sample of the given input (x) test data points.
        Underlying probability for picking each data point is predefined by global_param.p_nystroem_ratio.
        :return: tf.Tensor of shape [self.n_inducting_test, 1]
        """
        self.inducting_x_test = tf.gather(self.data_x_test, indices)

        return self.inducting_x_test

    def get_x_range(self) -> List[List[float]]:
        """
        Gives the range of all the input data points (training and test data).
        E.g. for min-max-scaled data on the interval of [0;1], this method retrieves the following float array: [0,1]
        :return: float array corresponding to the input (x) range of the data set
        """
        x_range = []

        for d in range(self.get_input_dimensionality()):
            p_min = float(min([min(self.data_x_train[:, d]), min(self.data_x_test[:, d])]))
            p_max = float(max([max(self.data_x_train[:, d]), max(self.data_x_test[:, d])]))
            x_range.append([p_min, p_max])

        return x_range

    def get_detrended_y_train(self) -> tf.Tensor:
        """
        Retrieves a detrended instantiation of the training target data (y train).
        Detrending simply means to subtract the mean functions value at every X from the corresponding y,
            i.e. self.data_y_train - self.mean_function(self.data_x_train)
        :return: tf.Tensor of shape [self.n_train, 1] or None if self.mean_function is not set
        """
        if self.mean_function is not None:
            if self.detrended_y_train is None:
                if isinstance(self.mean_function, bmf.ZeroMeanFunction):
                    self.detrended_y_train = tf.constant(self.data_y_train, dtype=global_param.p_dtype)
                else:
                    mean = tf.reshape(self.mean_function.get_tf_tensor(
                        self.mean_function.get_last_hyper_parameter(), self.data_x_train), shape=[-1, 1])
                    self.detrended_y_train = \
                        tf.subtract(self.data_y_train,
                                    mean)
            return self.detrended_y_train
        else:
            logging.error("Mean Function is None.")
            return None

    def get_detrended_y_test(self) -> tf.Tensor:
        if self.mean_function is not None:
            if self.detrended_y_test is None:
                self.detrended_y_test = self.get_detrended_y_test_individual(self.data_x_test, self.data_y_test)
            return self.detrended_y_test
        else:
            logging.error("Mean Function is None.")
            return None

    def get_detrended_y_test_individual(
            self, data_x_test: np.ndarray, data_y_test: np.ndarray) -> tf.Tensor:
        """
        Retrieves a detrended instantiation of the test target data (y test).
        Detrending simply means to subtract the mean functions value at every X from the corresponding y,
            i.e. self.data_y_test - self.mean_function(self.data_x_test)
        :return: tf.Tensor of shape [self.n_test, 1] or None if self.mean_function is not set
        """
        if isinstance(self.mean_function, bmf.ZeroMeanFunction):
            detrended_y_test = tf.constant(data_y_test, dtype=global_param.p_dtype)
        else:
            mean = tf.reshape(self.mean_function.get_tf_tensor(
                self.mean_function.get_last_hyper_parameter(), data_x_test), shape=[-1, 1])
            detrended_y_test = \
                tf.subtract(data_y_test, mean)

        return detrended_y_test

    def get_random_subset(self, subset_size: int):
        separate_train_test: bool = not np.array_equal(self.data_x_train, self.data_x_test)
        random_train_indices = tf.sort(tf.random.stateless_uniform(
            [subset_size, ], minval=0, maxval=self.n_train, dtype=tf.int64, seed=[self.seed, 1]))

        subset_x_train = tf.gather(self.data_x_train, random_train_indices)
        subset_y_train = tf.gather(self.data_y_train, random_train_indices)

        if separate_train_test:
            subset_x_test = self.data_x_test
            subset_y_test = self.data_y_test
        else:
            subset_x_test = self.data_x_train
            subset_y_test = self.data_y_train

        data_input = DataInput(subset_x_train, subset_y_train, subset_x_test, subset_y_test)

        data_input.set_mean_function(self.mean_function)

        return data_input

    def get_grid_subset(self, subset_size: int):
        separate_train_test: bool = not np.array_equal(self.data_x_train, self.data_x_test)
        random_train_indices = tf.constant(
            np.linspace(start=0, stop=self.n_train, num=subset_size, endpoint=False, dtype=int),
            dtype=tf.int64, shape=[subset_size, ])

        subset_x_train = tf.gather(self.data_x_train, random_train_indices)
        subset_y_train = tf.gather(self.data_y_train, random_train_indices)

        if separate_train_test:
            subset_x_test = self.data_x_test
            subset_y_test = self.data_y_test
        else:
            subset_x_test = self.data_x_train
            subset_y_test = self.data_y_train

        data_input = DataInput(subset_x_train, subset_y_train, subset_x_test, subset_y_test)

        data_input.set_mean_function(self.mean_function)

        return data_input

    def is_equidistant_input_x(self) -> bool:
        return is_equidistant(self.data_x_train)

    def get_subset(self, subset_size: int, subset_of_data_approach: mht.SubsetOfDataApproaches):
        if subset_of_data_approach is mht.SubsetOfDataApproaches.SOD_GRID:
            return self.get_grid_subset(subset_size)
        elif subset_of_data_approach is mht.SubsetOfDataApproaches.SOD_RANDOM:
            return self.get_random_subset(subset_size)
        else:
            raise Exception("Invalid subset-of-data approach: %s" % str(subset_of_data_approach))

    @staticmethod
    def get_k_fold_data_inputs(data_x_train: tf.Tensor, data_y_train: tf.Tensor, k: int, seed: int = 3061941):
        assert len(data_x_train.shape) == 2 and len(data_y_train.shape) == 2, "Only non-batched Data is valid Input."
        k_fold_data_inputs = AbstractDataInput.get_k_fold_data_inputs(data_x_train, data_y_train, k, seed)

        return [
            DataInput(data_input.data_x_train, data_input.data_y_train, data_input.data_x_test, data_input.data_y_test,
                      seed=data_input.seed)
            for data_input in k_fold_data_inputs]


class PartitionedDataInput(DataInput):
    def __init__(self, data_x_train: np.ndarray, data_y_train: np.ndarray, data_x_test: np.ndarray,
                 data_y_test: np.ndarray, data_inputs: List[DataInput]):
        super(PartitionedDataInput, self).__init__(data_x_train, data_y_train, data_x_test, data_y_test)
        self.data_inputs: List[DataInput] = data_inputs

    def set_mean_function(self, mean_function: mf.MeanFunction):
        """
        To allow for retrieving a detrended instantiation of the the target variables training_y and test_y,
        the mean function of the current Gaussian Process of interest may be set using this method.
        The mean function is also set for all segment-wise DataInput objects managed by this instantiation of
        BlockwiseDataInput.
        :param mean_function: to-be-set mean function
        """
        super(PartitionedDataInput, self).set_mean_function(mean_function)
        for data_input in self.data_inputs:
            data_input.set_mean_function(mean_function)


class BlockwiseDataInput(PartitionedDataInput):
    """
    Beyond offering all the functionality DataInput offers, BlockwiseDataInput also segments the given data according
    to given change points and computes the corresponding DataInput objects for all the resulting data segments.
    BlockwiseDataInput is especially used for globally segmented Gaussian Processes (i.e. BlockwiseGaussianProcess)
    """

    def __init__(self, data_x_train: np.ndarray, data_y_train: np.ndarray, data_x_test: np.ndarray,
                 data_y_test: np.ndarray, change_points: List[tf.Tensor]):
        """
        Beyond offering all the functionality DataInput offers, BlockwiseDataInput also segments the given data according
        to given change points and computes the corresponding DataInput objects for all the resulting data segments.
        BlockwiseDataInput is especially used for globally segmented Gaussian Processes (i.e. BlockwiseGaussianProcess)
        :param data_x_train:
        :param data_y_train:
        :param data_x_test:
        :param data_y_test:
        :param change_points:
        """
        data_inputs: List[DataInput] = []

        for i in range(0, len(change_points) + 1):
            if i == 0:
                train_idx = tf.where(data_x_train < tf.reshape(change_points[i], []))[:, 0]
                test_idx = tf.where(data_x_test < tf.reshape(change_points[i], []))[:, 0]
            elif i == len(change_points):
                train_idx = tf.where(data_x_train >= tf.reshape(change_points[i - 1], []))[:, 0]
                test_idx = tf.where(data_x_test >= tf.reshape(change_points[i - 1], []))[:, 0]
            else:
                train_idx = tf.where(
                    tf.math.logical_and(data_x_train < tf.reshape(change_points[i], []),
                                        data_x_train >= tf.reshape(change_points[i - 1], [])))[:, 0]
                test_idx = tf.where(
                    tf.math.logical_and(data_x_test < tf.reshape(change_points[i], []),
                                        data_x_test >= tf.reshape(change_points[i - 1], [])))[:, 0]

            block_data_x_train = tf.gather(data_x_train, train_idx)
            block_data_y_train = tf.gather(data_y_train, train_idx)
            block_data_x_test = tf.gather(data_x_test, test_idx)
            block_data_y_test = tf.gather(data_y_test, test_idx)

            data_inputs.append(DataInput(block_data_x_train, block_data_y_train, block_data_x_test, block_data_y_test))

        super(BlockwiseDataInput, self).__init__(data_x_train, data_y_train, data_x_test, data_y_test, data_inputs)
