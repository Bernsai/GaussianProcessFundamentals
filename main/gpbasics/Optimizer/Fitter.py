import gpbasics.global_parameters as global_param
from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput

global_param.ensure_init()

import logging
from typing import Tuple, List, Union

import tensorflow as tf
import tensorflow_probability as tfp

import gpbasics.Metrics.MatrixHandlingTypes as mht

import gpbasics.Optimizer.FitterType as ft
import gpbasics.Statistics.GaussianProcess as gp
import gpbasics.Metrics.Metrics as met
import gpbasics.Metrics.Auxiliary as met_aux


class Fitter:
    def __init__(self, data_input: Union[AbstractDataInput, List[AbstractDataInput]],
                 gaussian_process: gp.AbstractGaussianProcess, metric_type: met.MetricType,
                 fitter_type: ft.FitterType, from_distribution: bool, local_approx: mht.GlobalApproximationsType,
                 numerical_matrix_handling: mht.NumericalMatrixHandlingType, subset_size: int = None):
        self._gp: gp.AbstractGaussianProcess = gaussian_process
        self.metric: Union[met.Metric, List[met.Metric]]
        if isinstance(data_input, list):
            self.metric = []
            for data_input_instance in data_input:
                copied_gp: gp.AbstractGaussianProcess = self._gp.copy()
                copied_gp.set_data_input(data_input_instance)
                self.metric.append(met_aux.get_metric_by_type(
                    metric_type, copied_gp, local_approx, numerical_matrix_handling, subset_size))
        else:
            self._gp.set_data_input(data_input)
            self.metric = met_aux.get_metric_by_type(
                metric_type, self._gp, local_approx, numerical_matrix_handling, subset_size)

        self.data_input: Union[AbstractDataInput, List[AbstractDataInput]] = data_input
        self.type = fitter_type
        self.from_distribution: bool = from_distribution

    def fit(self) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor], tf.Tensor, tf.Tensor]:
        pass


class GradientFitter(Fitter):
    pass


class VariationalSgdFitter(Fitter):
    def __init__(
            self, data_input: Union[AbstractDataInput, List[AbstractDataInput]],
            gaussian_process: gp.AbstractGaussianProcess, metric_type: met.MetricType, from_distribution: bool,
            local_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
            subset_size: int = None):
        super(VariationalSgdFitter, self).__init__(
            data_input, gaussian_process, metric_type, ft.FitterType.NON_GRADIENT, from_distribution,
            local_approx, numerical_matrix_handling, subset_size)

    def fit(self) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor], tf.Tensor, tf.Tensor]:
        xrange: List[List[float]]
        n: int
        if isinstance(self.data_input, list):
            xrange = self.data_input[0].get_x_range()
            n = self.data_input[0].n_train
        else:
            xrange = self.data_input.get_x_range()
            n = self.data_input.n_train

        default_noise = tf.Variable(global_param.p_cov_matrix_jitter, dtype=global_param.p_dtype)

        kernel = self._gp.covariance_matrix.kernel
        hyper_parameter: List[tf.Tensor] = kernel.get_default_hyper_parameter(xrange, n, self.from_distribution)

        if isinstance(self.metric, list) and (self.metric[0].local_approx is mht.MatrixApproximations.SKC_LOWER_BOUND or \
                self.metric[0].local_approx is mht.MatrixApproximations.BASIC_NYSTROEM):
            indices = tf.Variable(tf.random.stateless_uniform(
                [self.data_input[0].n_inducting_train, 1], minval=0, maxval=1,
                dtype=tf.float64, seed=[self.data_input[0].seed, 1]))

        elif isinstance(self.metric, met.AbstractMetric) and \
                (self.metric.local_approx is mht.MatrixApproximations.SKC_LOWER_BOUND or \
                self.metric.local_approx is mht.MatrixApproximations.BASIC_NYSTROEM):
            indices = tf.Variable(tf.random.stateless_uniform(
                [self.data_input.n_inducting_train, 1], minval=0, maxval=1,
                dtype=tf.float64, seed=[self.data_input.seed, 1]))
        else:
            indices = None

        def opt():
            return self.metric.get_metric(hyper_parameter, default_noise, indices)

        def opt_optimize_noise():
            return self.metric.get_metric(hyper_parameter[1:], tf.abs(hyper_parameter[:1][0]), indices)

        def opt_kfold():
            return tf.reduce_mean([m.get_metric(hyper_parameter, default_noise, indices) for m in self.metric])

        def opt_optimize_noise_kfold():
            return tf.reduce_mean([m.get_metric(hyper_parameter[1:], tf.abs(hyper_parameter[:1][0], indices))
                                   for m in self.metric])

        sgd_opt: tfp.optimizer.VariationalSGD = \
            tfp.optimizer.VariationalSGD(batch_size=max(10, int(n / 100)), total_num_examples=n, burnin=int(n / 10))

        if global_param.p_optimize_noise:
            hyper_parameter = [default_noise] + hyper_parameter

            if isinstance(self.metric, list):
                opt_func = opt_optimize_noise_kfold
            else:
                opt_func = opt_optimize_noise
        else:
            if isinstance(self.metric, list):
                opt_func = opt_kfold
            else:
                opt_func = opt

        pre_fit_metric: tf.Tensor = opt_func()

        if global_param.p_check_hyper_parameters:
            bounds = self._gp.kernel.get_hyper_parameter_bounds(xrange, n)
            with tf.GradientTape() as g:
                for h in hyper_parameter:
                    g.watch(h)

                if indices is not None:
                    g.watch(indices)
                    gradients = g.gradient(opt_func(), hyper_parameter + [indices])
                else:
                    gradients = g.gradient(opt_func(), hyper_parameter)

            def gradient_bounding(idx):
                if hyper_parameter[idx] < bounds[idx][0]:
                    g_base = tf.abs((bounds[idx][0] / hyper_parameter[idx]))
                    return tf.cast(-g_base, dtype=global_param.p_dtype)
                elif hyper_parameter[idx] > bounds[idx][1]:
                    g_base = tf.abs((hyper_parameter[idx] / bounds[idx][1]))
                    return tf.cast(g_base, dtype=global_param.p_dtype)

                return gradients[idx]

            gradients = list(map(gradient_bounding, list(range(len(bounds)))))

            if indices is not None:
                grads_and_vars = [(g, h) for g, h in zip(gradients[:-1], hyper_parameter)]
                grads_and_vars += [(gradients[-1:][0], indices)]
            else:
                grads_and_vars = [(g, h) for g, h in zip(gradients, hyper_parameter)]

            sgd_opt.apply_gradients(grads_and_vars)

        else:
            if indices is not None:
                sgd_opt.minimize(opt_func, hyper_parameter + [indices])
            else:
                sgd_opt.minimize(opt_func, hyper_parameter)

        if global_param.p_optimize_noise:
            kernel.set_last_hyper_parameter(hyper_parameter[1:])
            kernel.set_noise(tf.abs(hyper_parameter[:1][0]))
        else:
            kernel.set_last_hyper_parameter(hyper_parameter)
            kernel.set_noise(global_param.p_cov_matrix_jitter)

        post_fit_metric: tf.Tensor = opt_func()

        return pre_fit_metric, post_fit_metric, self._gp.covariance_matrix.kernel.get_last_hyper_parameter(), \
               self._gp.covariance_matrix.kernel.get_noise(), indices


if global_param.p_gradient_fitter is None:
    global_param.p_gradient_fitter = VariationalSgdFitter
    if __name__ == '__main__':
        logging.info("Gradient Fitter set to default: %s" % global_param.p_gradient_fitter)

if global_param.p_non_gradient_fitter is None:
    global_param.p_non_gradient_fitter = None
    if __name__ == '__main__':
        logging.info("Non-Gradient Fitter set to default: %s" % global_param.p_non_gradient_fitter)
