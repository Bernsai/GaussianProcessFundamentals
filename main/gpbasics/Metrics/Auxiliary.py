import logging

from gpbasics.Metrics import MatrixHandlingTypes as mht
from gpbasics.Metrics.BayesianInformationCriterion import BIC, BlockwiseBIC
from gpbasics.Metrics.LogLikelihood import LogLikelihood, BlockwiseLogLikelihood
from gpbasics.Metrics.MeanSquaredError import MeanSquaredError, BlockwiseMeanSquaredError
from gpbasics.Metrics.Metrics import MetricType, Metric
from gpbasics.Metrics.SkcLogLikelihood import LogLikelihoodUpperBound
from gpbasics.Statistics import Nystroem_K as ny
from gpbasics.Statistics import GaussianProcess as gp


def get_metric_by_type(
        metric_type: MetricType, _gp: gp.AbstractGaussianProcess,
        local_approx: mht.GlobalApproximationsType = mht.MatrixApproximations.NONE,
        numerical_matrix_handling: mht.NumericalMatrixHandlingType = mht.NumericalMatrixHandlingType.CHOLESKY_BASED,
        subset_size: int = None) -> Metric:
    if metric_type is MetricType.LL:
        if local_approx is mht.MatrixApproximations.SKC_UPPER_BOUND:
            nyk = ny.NystroemMatrix(_gp.covariance_matrix)
            nyk.set_data_input(_gp.data_input)
            return LogLikelihoodUpperBound(_gp.data_input, _gp.covariance_matrix, nystroem_k=nyk)
        else:
            return LogLikelihood(
                _gp.data_input, _gp.covariance_matrix, local_approx, numerical_matrix_handling, subset_size)

    if metric_type is MetricType.BIC:
        ll = LogLikelihood(_gp.data_input, _gp.covariance_matrix, local_approx, numerical_matrix_handling, subset_size)
        return BIC(_gp.data_input, _gp.covariance_matrix, ll)

    if metric_type is MetricType.MSE:
        return MeanSquaredError(
            _gp.data_input, _gp.covariance_matrix, _gp.aux, local_approx, numerical_matrix_handling, subset_size)

    if metric_type is MetricType.blockwise_MSE:
        assert isinstance(_gp, gp.BlockwiseGaussianProcess) or isinstance(_gp, gp.PartitionedGaussianProcess), \
            "Blockwise MSE may only be determined for blockwise Gaussian Process."
        return BlockwiseMeanSquaredError(_gp, local_approx, numerical_matrix_handling, subset_size)

    if metric_type is MetricType.blockwise_BIC:
        assert isinstance(_gp, gp.BlockwiseGaussianProcess) or isinstance(_gp, gp.PartitionedGaussianProcess), \
            "Blockwise BIC may only be determined for blockwise Gaussian Process."
        return BlockwiseBIC(_gp, local_approx, numerical_matrix_handling, subset_size)

    if metric_type is MetricType.blockwise_LL:
        assert isinstance(_gp, gp.BlockwiseGaussianProcess) or isinstance(_gp, gp.PartitionedGaussianProcess), \
            "Blockwise Log Likelihood may only be determined for blockwise Gaussian Process."
        return BlockwiseLogLikelihood(_gp, local_approx, numerical_matrix_handling, subset_size)

    logging.error("Invalid MetricType: %s" % str(metric_type))
    return None


def get_blockwise_metric_for_standard_metric(metric_type: MetricType) -> MetricType:
    if metric_type.value >= 10:
        logging.warning("get_blockwise_metric_for_standard_metric received blockwise metric and thus had no effect.")
    else:
        if metric_type is MetricType.LL:
            return MetricType.blockwise_LL
        elif metric_type is MetricType.BIC:
            return MetricType.blockwise_BIC
        elif metric_type is MetricType.MSE:
            return MetricType.blockwise_MSE
        else:
            logging.warning("There is no blockwise version for metric %s." % str(metric_type))
    return metric_type
