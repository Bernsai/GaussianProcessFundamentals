import gpbasics.global_parameters as global_param
global_param.ensure_init()

import tensorflow as tf
import gpbasics.Optimizer.FitterType as ft
import gpbasics.Statistics.GaussianProcess as gp
from gpbasics.Metrics import Metrics as met
from gpbasics.Optimizer import Fitter as f
from typing import Tuple, List
from scipy.optimize import line_search


class FletcherReevesCgFitter(f.GradientFitter):
    def __init__(self, gaussian_process: gp.AbstractGaussianProcess, metric_type: met.MetricType):
        super(FletcherReevesCgFitter, self).__init__(gaussian_process, metric_type, ft.FitterType.GRADIENT)

    def fit(self) -> Tuple[tf.Tensor, tf.Tensor]:
        xrange: List[List[float]] = self._gp.data_input.get_x_range()
        n: int = self._gp.data_input.n_train
        k = 0

        def objective_function(serialized_hyper_parameters: tf.Tensor):
            deserialized_hyp: List[tf.Tensor] = kernel.deserialize_hyper_parameter(
                serialized_hyper_parameters, kernel.get_hyper_parameter_dimensionalities())
            metric = self.metric.get_metric(deserialized_hyp)
            return metric

        def gradient_function(serialized_hyper_parameters: tf.Tensor):
            deserialized_hyp: List[tf.Tensor] = kernel.deserialize_hyper_parameter(
                serialized_hyper_parameters, kernel.get_hyper_parameter_dimensionalities())
            return self.metric.get_gradients(deserialized_hyp)

        kernel = self._gp.covariance_matrix.kernel
        x: List[tf.Tensor] = [kernel.serialize_hyper_parameter(kernel.get_default_hyper_parameter(xrange, n))]

        f: List[tf.Tensor] = [objective_function(x[k])]

        grad_f: List[tf.Tensor] = [gradient_function(x[k])]

        p: List[tf.Tensor] = [tf.cast(-1, dtype=global_param.p_dtype) * grad_f[k]]

        while tf.abs(tf.reduce_max(grad_f[k])) > 1e-4:
            result = line_search(f=objective_function, myfprime=gradient_function, xk=x[k].numpy(),
                                 pk=p[k].numpy(), maxiter=10000)

            if result[0] is None:
                raise Exception("Line Search did not converge.")
            else:
                x.append(x[k] + tf.cast(result[0], dtype=global_param.p_dtype) * p[k])
                grad_f.append(gradient_function(x[k + 1]))
                next_beta_k = self.get_next_beta_k(grad_f, k)
                p.append(tf.cast(-1, dtype=global_param.p_dtype) * grad_f[k + 1] + next_beta_k * p[k])

                k = k + 1

                if k - 1 >= 0:
                    p[k - 1] = None
                    grad_f[k - 1] = None
                    x[k - 1] = None

            print(result)

        return f0, post_fit_metric

    def get_next_beta_k(self, grad_f, k):
        if tf.shape(grad_f[k + 1])[0] > 1:
            numerator = tf.matmul(tf.transpose(grad_f[k + 1]), grad_f[k + 1])
        else:
            numerator = tf.square(grad_f[k + 1])

        if tf.shape(grad_f[k])[0] > 1:
            denominator = tf.matmul(tf.transpose(grad_f[k]), grad_f[k])
        else:
            denominator = tf.square(grad_f[k])
        return tf.divide(numerator, denominator)
