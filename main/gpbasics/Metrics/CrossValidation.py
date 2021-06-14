import typing
import gpbasics.global_parameters as global_param
global_param.ensure_init()

import gpbasics.Statistics.GaussianProcess as gp
import gpbasics.DataHandling.DataInput as di
import gpbasics.Metrics.Metrics as met
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.Optimizer.Fitter as f
import gpbasics.Metrics.Auxiliary as met_aux
import numpy as np
from typing import List
import tensorflow as tf


def get_data_inputs(data_input: di.DataInput, test_ratio: float = 0.2):
    n_samples = data_input.n_train
    idx_test_samples = 0
    n_test_samples = int(round(n_samples * test_ratio))
    n_cross_valid_epochs = int(np.floor(1 / test_ratio))

    # Endpoint false means
    indices = np.linspace(start=0, num=n_samples, stop=n_samples, endpoint=False, dtype=int)

    np.random.shuffle(indices)

    data_inputs: typing.List[di.DataInput] = []

    for i in range(n_cross_valid_epochs):
        stop_idx_test = (n_test_samples + idx_test_samples)
        test_indices = sorted(indices[idx_test_samples:stop_idx_test])
        train_indices = sorted(np.concatenate([indices[:idx_test_samples], indices[stop_idx_test:]]))
        x_train = tf.gather(data_input.data_x_train, train_indices)
        x_test = tf.gather(data_input.data_x_train, test_indices)
        y_train = tf.gather(data_input.data_y_train, train_indices)
        y_test = tf.gather(data_input.data_y_train, test_indices)

        perm_data_input = di.DataInput(x_train, y_train, x_test, y_test)
        perm_data_input.set_mean_function(data_input.mean_function)
        data_inputs.append(perm_data_input)

        idx_test_samples += n_test_samples

    return data_inputs


class CrossValidation:
    def __init__(self, gaussian_process: gp.AbstractGaussianProcess, data_input: di.DataInput,
                 local_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 subset_size: int = None, metric_type: met.MetricType = met.MetricType.MSE, random_restarts: int = 1):
        self.metric_type: met.MetricType = metric_type

        self.local_approx: mht.GlobalApproximationsType = local_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size

        if self.local_approx is not mht.MatrixApproximations.NONE and self.subset_size is None:
            self.subset_size = int(data_input.n_train * global_param.p_nystroem_ratio)

        if isinstance(gaussian_process, gp.GaussianProcess):
            self.gaussian_process = gp.GaussianProcess(
                gaussian_process.kernel.deepcopy(), gaussian_process.mean_function.deepcopy())
        else:
            self.gaussian_process = gp.BlockwiseGaussianProcess(
                gaussian_process.kernel.deepcopy(), gaussian_process.mean_function.deepcopy())

        self.data_input = data_input

        self.random_restarts: int = max(1, random_restarts)

    def cross_validation(self, test_ratio: float = 0.2):
        data_inputs: typing.List[di.DataInput]
        if self.metric_type.value >= 10 and isinstance(self.data_input, di.PartitionedDataInput) and \
                (isinstance(self.gaussian_process, gp.BlockwiseGaussianProcess)
                 or isinstance(self.gaussian_process, gp.PartitionedGaussianProcess)):
            data_inputs = self.get_partitioned_data_inputs(self.data_input, test_ratio)
        else:
            data_inputs = get_data_inputs(self.data_input, test_ratio)

        list_of_metric_results: typing.List[float] = []

        for data_input in data_inputs:
            self.gaussian_process.set_data_input(data_input)

            metric = met_aux.get_metric_by_type(
                self.metric_type, self.gaussian_process, local_approx=self.local_approx,
                numerical_matrix_handling=self.numerical_matrix_handling, subset_size=self.subset_size)

            metric_result = metric.get_metric(
                self.gaussian_process.kernel.get_last_hyper_parameter(), self.gaussian_process.kernel.get_noise())

            list_of_metric_results.append(float(metric_result.numpy()))

        return np.mean(list_of_metric_results)

    def get_partitioned_data_inputs(self, partitioned_data_input: di.PartitionedDataInput, test_ratio: float = 0.2):
        data_inputs: typing.List[typing.List[di.DataInput]] = []

        partitioned_data_inputs: typing.List[di.PartitionedDataInput] = []

        n_data_inputs = -1

        for data_input in partitioned_data_input.data_inputs:
            inputs = get_data_inputs(data_input, test_ratio)
            if n_data_inputs == -1:
                n_data_inputs = len(inputs)
            else:
                assert len(inputs) == n_data_inputs, \
                    "All data_inputs of partitioned data inputs need to have the same amount of permutations"
            data_inputs.append(inputs)

        for i in range(n_data_inputs):
            permutated_inputs = [permutations[i] for permutations in data_inputs]

            data_x_trains: List[np.ndarray] = []
            data_y_trains: List[np.ndarray] = []
            data_x_tests: List[np.ndarray] = []
            data_y_tests: List[np.ndarray] = []

            for p_input in permutated_inputs:
                data_x_trains.append(p_input.data_x_train)
                data_x_tests.append(p_input.data_x_test)
                data_y_trains.append(p_input.data_y_train)
                data_y_tests.append(p_input.data_y_test)

            perm_data_input = di.PartitionedDataInput(np.concatenate(data_x_trains), np.concatenate(data_y_trains),
                                                      np.concatenate(data_x_tests), np.concatenate(data_y_tests),
                                                      permutated_inputs)

            perm_data_input.set_mean_function(self.data_input.mean_function)

            partitioned_data_inputs.append(perm_data_input)

        return partitioned_data_inputs
