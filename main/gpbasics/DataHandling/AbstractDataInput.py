import gpbasics.global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf
import numpy as np
import gpbasics.MeanFunctionBasics.MeanFunction as mf
import logging
from typing import List
import gpbasics.KernelBasics.Kernel as k
import gpbasics.Metrics.MatrixHandlingTypes as mht


class AbstractDataInput:
    def __init__(self, data_x_train: tf.Tensor, data_y_train: tf.Tensor, data_x_test: tf.Tensor = None,
                 data_y_test: tf.Tensor = None, test_ratio: float = -1, seed: int = 3061941):
        assert (data_x_test is None or (len(data_x_train.shape) == len(data_x_test.shape) and len(data_y_train.shape) == len(data_y_test.shape) \
               and len(data_x_train.shape) == len(data_y_train.shape))) and \
               (len(data_x_train.shape) == 2 or len(data_x_train.shape) == 3), \
            "Shape of input and target data (test as well as train data) needs to be either " \
            "[instance#, length, dimensionality] or [length, dimensionality]."
        assert data_y_train.shape[len(data_y_train.shape) - 1] == 1, \
            "Target training data (data_y_train) has to be unidimensional, shape=[n_train, 1] / [instance#, n_train, 1]"
        assert data_y_test is None or data_y_test.shape[len(data_y_test.shape) - 1] == 1, \
            "Target test data (data_y_test) has to be unidimensional, shape=[n_test, 1] / [instance#, n_test, 1]"
        assert data_x_test is None or data_x_train.shape[-1] == data_x_test.shape[-1], \
            "Dimensionality of training and test input data (data_x_train and data_x_test) need to match"

        assert test_ratio <= 1, "test_ratio has to be in the range [0; 1]"

        self.seed: int = seed

        if len(data_x_train.shape) == 2:
            logging.debug("Non-Batch Data-Input used.")
        elif len(data_x_train.shape) == 3:
            logging.debug("Batch Data-Input used.")

        if test_ratio > 0 and data_x_test is not None:
            logging.warning("test_ratio is ignored if test_data is explicitly given.")

        if data_x_test is None and test_ratio != 0:
            if test_ratio < 0:
                logging.warning("test_ratio is not given although explicit test data was not provided. "
                                "default value '0.2' is assumed for test_ratio.")
                test_ratio = 0.2

            length = data_x_train.shape[0]
            test_size = min(length - 1, int(length * test_ratio))
            train_size = length - test_size

            shuffled_idx = tf.random.shuffle(tf.cast(tf.linspace(0, length-1, length), dtype=tf.int32), seed=self.seed)
            idx_train, idx_test = tf.split(shuffled_idx, [train_size, test_size])

            idx_train = tf.sort(idx_train)
            idx_test = tf.sort(idx_test)

            self.data_x_train: tf.Tensor = tf.gather(data_x_train, idx_train, axis=0)
            self.data_y_train: tf.Tensor = tf.gather(data_y_train, idx_train, axis=0)
            self.data_x_test: tf.Tensor = tf.gather(data_x_train, idx_test, axis=0)
            self.data_y_test: tf.Tensor = tf.gather(data_y_train, idx_test, axis=0)

        elif data_x_test is None:
            self.data_x_train: tf.Tensor = data_x_train
            self.data_y_train: tf.Tensor = data_y_train
            self.data_x_test: tf.Tensor = data_x_train
            self.data_y_test: tf.Tensor = data_y_train

        else:
            self.data_x_train: tf.Tensor = data_x_train
            self.data_y_train: tf.Tensor = data_y_train
            self.data_x_test: tf.Tensor = data_x_test
            self.data_y_test: tf.Tensor = data_y_test

        self.detrended_y_test: tf.Tensor = None
        self.detrended_y_train: tf.Tensor = None
        self.mean_function: mf.MeanFunction = None

        self.n_train: int = self.data_x_train.shape[-2]
        self.n_test: int = self.data_x_test.shape[-2]
        self.inducting_x_train: tf.Tensor = None
        self.inducting_x_test: tf.Tensor = None
        inducting_min: int = 20
        self.n_inducting_train: int = max(inducting_min, int(self.n_train * global_param.p_nystroem_ratio))
        self.n_inducting_test: int = max(inducting_min, int(self.n_test * global_param.p_nystroem_ratio))

    def get_input_dimensionality(self) -> int:
        """
        Returns the number of dimensions of the input data, i.e. data_x_train and data_x_test
        :return:
        """
        return int(self.data_x_train.shape[len(self.data_x_train.shape) - 1])

    def set_seed(self, seed: int):
        """
        Sets the seed used for retrieving uniform random sample of the given input (x) data;
        used by methods 'get_inducting_x_train' & 'get_inducting_x_test'
        :param seed: new seed to be set
        :return:
        """
        self.seed = seed
        self.inducting_x_test = None
        self.inducting_x_train = None

    def set_mean_function(self, mean_function: mf.MeanFunction):
        """
        To allow for retrieving a detrended instantiation of the the target variables training_y and test_y,
        the mean function of the current Gaussian Process of interest may be set using this method.
        :param mean_function: to-be-set mean function
        """
        self.detrended_y_train = None
        self.detrended_y_test = None
        self.mean_function = mean_function

        if self.mean_function.get_last_hyper_parameter() is None:
            self.mean_function.last_hyper_parameter = self.mean_function.get_default_hyper_parameter()

    def get_inducting_x_train(self, indices) -> tf.Tensor:
        pass

    def get_inducting_x_test(self, indices) -> tf.Tensor:
        pass

    def get_x_range(self) -> List[List[float]]:
        pass

    def get_detrended_y_train(self) -> tf.Tensor:
        pass

    def get_detrended_y_test(self) -> tf.Tensor:
        pass

    def get_random_subset(self, subset_size: int):
        pass

    def get_grid_subset(self, subset_size: int):
        pass

    def get_independent_smoothed_grid_subset(self, subset_size: int, smoothing_kernel: k.Kernel = None):
        pass

    def is_equidistant_input_x(self) -> bool:
        pass

    def get_subset(self, subset_size: int, subset_of_data_approach: mht.SubsetOfDataApproaches):
        pass

    @staticmethod
    def get_k_fold_data_inputs(x_train: tf.Tensor, y_train: tf.Tensor, k: int, seed: int = 3061941):
        length = x_train.shape[0]
        all_idx = tf.cast(tf.linspace(0, length - 1, length), dtype=tf.int32)
        shuffled_idx = tf.random.shuffle(all_idx, seed=seed)
        k_list = [length // k] * (k - 1)
        k_list += [length - ((length // k) * (k - 1))]
        list_idx = tf.split(shuffled_idx, k_list)

        data_inputs: List[AbstractDataInput] = []

        for i in range(k):
            idx_test = tf.sort(tf.constant(list_idx[i], tf.int32))
            idx_train = tf.sort(tf.concat([list_idx[j] for j in range(k) if j != i], axis=0))
            data_x_train: tf.Tensor = tf.gather(x_train, idx_train, axis=0)
            data_y_train: tf.Tensor = tf.gather(y_train, idx_train, axis=0)
            data_x_test: tf.Tensor = tf.gather(x_train, idx_test, axis=0)
            data_y_test: tf.Tensor = tf.gather(y_train, idx_test, axis=0)

            data_inputs.append(AbstractDataInput(data_x_train, data_y_train, data_x_test, data_y_test, seed=seed))

        return data_inputs