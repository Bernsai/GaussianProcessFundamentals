import gpbasics.global_parameters as global_param

global_param.ensure_init()

from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput
import tensorflow as tf
import logging
from typing import List
import gpbasics.KernelBasics.Kernel as k
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.MeanFunctionBasics.BaseMeanFunctions as bmf


def is_equidistant(input_data: tf.Tensor) -> bool:
    length = input_data.shape[1]
    diff = input_data[:, :length-1, :] - input_data[:, 1:, :]
    mean = tf.reduce_mean(diff, axis=0)
    diff_max = tf.abs(tf.reduce_max(diff, axis=1) - mean)
    diff_min = tf.abs(tf.reduce_min(diff, axis=1) - mean)
    allowed_error = tf.constant(1 / (100 * length), dtype=global_param.p_dtype)
    return tf.logical_and(tf.less(diff_max, allowed_error), tf.less(diff_min, allowed_error))


class BatchDataInput(AbstractDataInput):
    def __init__(
            self, data_x_train: tf.Tensor, data_y_train: tf.Tensor, data_x_test: tf.Tensor = None,
            data_y_test: tf.Tensor = None, test_ratio: float = -1, seed: int = 3061941):
        super(BatchDataInput, self).__init__(data_x_train, data_y_train, data_x_test, data_y_test, test_ratio, seed)

    def get_inducting_x_train(self) -> tf.Tensor:
        raise Exception("get_inducting_x_train -- Not implemented for BatchDataInput.")

    def get_inducting_x_test(self) -> tf.Tensor:
        raise Exception("get_inducting_x_test -- Not implemented for BatchDataInput.")

    def get_x_range(self) -> List[List[float]]:
        x_range = []

        for d in range(self.get_input_dimensionality()):
            p_min = float(tf.reduce_min(
                tf.reduce_min(
                    tf.concat([self.data_x_train[:, :, d], self.data_x_test[:, :, d]], axis=0), axis=0), axis=0)) # float(min([min(self.data_x_train[:, d]), min(self.data_x_test[:, d])]))
            p_max = float(tf.reduce_max(
                tf.reduce_max(
                    tf.concat([self.data_x_train[:, :, d], self.data_x_test[:, :, d]], axis=0), axis=0), axis=0))
            x_range.append([p_min, p_max])

        return x_range

    def get_detrended_y_train(self) -> tf.Tensor:
        if self.mean_function is not None:
            if self.detrended_y_train is None:
                if isinstance(self.mean_function, bmf.ZeroMeanFunction):
                    self.detrended_y_train = self.data_y_train
                else:
                    mean_function = self.mean_function

                    def get_mean(x_train):
                        return tf.reshape(self.mean_function.get_tf_tensor(
                            mean_function.get_last_hyper_parameter(), x_train), shape=[-1, 1])

                    mean = tf.map_fn(get_mean, self.data_x_train)
                    self.detrended_y_train = tf.subtract(self.data_y_train, mean)
            return self.detrended_y_train
        else:
            logging.error("Mean Function is None.")
            return None

    def get_detrended_y_test(self) -> tf.Tensor:
        if self.mean_function is not None:
            if self.detrended_y_test is None:
                if isinstance(self.mean_function, bmf.ZeroMeanFunction):
                    self.detrended_y_test = self.data_y_test
                else:
                    mean_function = self.mean_function

                    def get_mean(x_test):
                        return tf.reshape(self.mean_function.get_tf_tensor(
                            mean_function.get_last_hyper_parameter(), x_test), shape=[-1, 1])

                    mean = tf.map_fn(get_mean, self.data_x_test)
                    self.detrended_y_train = tf.subtract(self.data_y_test, mean)
            return self.detrended_y_train
        else:
            logging.error("Mean Function is None.")
            return None

    def get_random_subset(self, subset_size: int):
        raise Exception("get_random_subset -- Not implemented for BatchDataInput.")

    def get_grid_subset(self, subset_size: int):
        raise Exception("get_grid_subset -- Not implemented for BatchDataInput.")

    def get_independent_smoothed_grid_subset(self, subset_size: int, smoothing_kernel: k.Kernel = None):
        raise Exception("get_independent_smoothed_grid_subset -- Not implemented for BatchDataInput.")

    def is_equidistant_input_x(self) -> bool:
        return is_equidistant(self.data_x_train)

    def get_subset(self, subset_size: int, subset_of_data_approach: mht.SubsetOfDataApproaches):
        raise Exception("get_subset -- Not implemented for BatchDataInput.")

    @staticmethod
    def get_k_fold_data_inputs(data_x_train: tf.Tensor, data_y_train: tf.Tensor, k: int, seed: int = 3061941):
        assert len(data_x_train.shape) == 3 and len(data_y_train.shape) == 3, "Only Batched Data is valid Input."
        k_fold_data_inputs = AbstractDataInput.get_k_fold_data_inputs(data_x_train, data_y_train, k, seed)

        return [
            BatchDataInput(
                data_input.data_x_train, data_input.data_y_train, data_input.data_x_test, data_input.data_y_test,
                seed=data_input.seed)
            for data_input in k_fold_data_inputs]
