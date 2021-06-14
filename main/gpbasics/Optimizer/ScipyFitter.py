from gpbasics import global_parameters as global_param

global_param.ensure_init()

from typing import Tuple, List

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from gpbasics.Metrics import Metrics as met
from gpbasics.Statistics import GaussianProcess as gp
from gpbasics.Optimizer import FitterType as ft
from gpbasics.Optimizer import Fitter


class ScipyBfgsFitter(Fitter):
    def __init__(self, gaussian_process: gp.AbstractGaussianProcess, metric_type: met.MetricType):
        super(ScipyBfgsFitter, self).__init__(gaussian_process, metric_type, ft.FitterType.NON_GRADIENT)

    def fit(self) -> Tuple[tf.Tensor, tf.Tensor]:
        xrange: List[List[float]] = self._gp.data_input.get_x_range()
        n: int = self._gp.data_input.n_train

        kernel = self._gp.covariance_matrix.kernel
        list_hyper_parameter: List[tf.Tensor] = kernel.get_default_hyper_parameter(xrange, n)
        hyper_parameter: np.ndarray = \
            kernel.serialize_hyper_parameter(list_hyper_parameter).numpy()

        dims = kernel.get_hyper_parameter_dimensionalities()

        pre_fit_metric: tf.Tensor = self.metric.get_metric(list_hyper_parameter)

        def opt(p_hyp: np.ndarray):
            deserialized_hyp = kernel.deserialize_hyper_parameter(
                tf.constant(p_hyp), dims)
            return self.metric.get_metric(deserialized_hyp).numpy()

        optimize_result = minimize(opt, hyper_parameter, method="BFGS")

        last_hyp = kernel.deserialize_hyper_parameter(tf.constant(optimize_result.x), dims)
        kernel.set_last_hyper_parameter(last_hyp)

        post_fit_metric: tf.Tensor = self.metric.get_metric(last_hyp)

        return pre_fit_metric, post_fit_metric
