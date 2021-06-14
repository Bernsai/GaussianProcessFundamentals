import logging
import multiprocessing
import os
import sys
from enum import Enum

import tensorflow as tf


class ChangePointOperatorType(Enum):
    SIGMOID = 0
    INDICATOR = 1
    APPROX_INDICATOR = 2


global p_cp_operator_type, p_cov_matrix_jitter, p_dtype, p_nystroem_ratio, p_used_base_kernel, \
    p_bk_per_default_init, p_bk_lin_default_init, p_bk_se_default_init, p_bk_c_default_init, p_split_kernel, \
    p_gradient_fitter, p_non_gradient_fitter, p_used_base_mean_functions, p_max_threads, p_logging_level, \
    pool, initiated, p_sub_model_depth_convergence, p_default_global_max_depth, p_default_local_max_depth, \
    p_default_npo, p_default_default_window_size, p_default_partitions_split_per_layer, p_scaled_base_kernel, \
    p_scale_data_y, p_batch_metric_aggregator, p_optimize_noise, p_check_hyper_parameters


def ensure_init():
    global initiated
    if not initiated:
        logging.warning("Global parameters not initiated!")
        sys.exit(-100)


def init(tf_parallel: int, worker: bool = False):
    global p_cp_operator_type, p_cov_matrix_jitter, p_dtype, p_nystroem_ratio, p_used_base_kernel, \
        p_split_kernel, p_gradient_fitter, p_non_gradient_fitter, p_used_base_mean_functions, \
        p_max_threads, p_logging_level, pool, initiated, p_sub_model_depth_convergence, p_default_global_max_depth, \
        p_default_local_max_depth, p_default_npo, p_default_default_window_size, p_default_partitions_split_per_layer, \
        p_scaled_base_kernel, p_scale_data_y, p_batch_metric_aggregator, p_optimize_noise, p_check_hyper_parameters

    tf.config.threading.set_inter_op_parallelism_threads(tf_parallel)
    tf.config.threading.set_intra_op_parallelism_threads(tf_parallel)

    initiated = True

    p_dtype = tf.float64
    p_cp_operator_type = ChangePointOperatorType.INDICATOR
    p_cov_matrix_jitter = tf.constant(1e-8, dtype=p_dtype)
    p_optimize_noise = False
    # SKC ideal: p_cov_matrix_jitter = tf.constant(1e-1, dtype=p_dtype)
    p_nystroem_ratio = 0.1
    p_check_hyper_parameters = False

    p_used_base_kernel = []
    p_used_base_mean_functions = []

    p_split_kernel = None
    p_gradient_fitter = None
    p_non_gradient_fitter = None

    p_max_threads = max([1, os.cpu_count() - tf_parallel])

    p_logging_level = logging.INFO

    p_scaled_base_kernel = False

    p_batch_metric_aggregator = tf.reduce_mean

    pool = None

    p_scale_data_y = True

    if __name__ == '__main__':
        logging.basicConfig(format='%(levelname)s: %(message)s', level=p_logging_level)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=p_logging_level)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logging.info("Process-%s:Initialization of global parameters finished." % os.getpid())


def set_up_pool(maxtasksperchild: int = -1):
    if p_max_threads > 1:
        global pool
        if maxtasksperchild is None or maxtasksperchild < 1:
            maxtasksperchild = None
        logging.info("Setting up pool of parallel workers.")
        pool = multiprocessing.Pool(processes=p_max_threads, maxtasksperchild=maxtasksperchild)
        logging.info("Pool of workers set up!")
    else:
        pool = None
        logging.warning("No Multiprocessing Pool set up, due to non-parallel execution")


def shutdown_pool():
    global pool
    if pool is not None:
        pool.close()
        pool.join()
        pool = None
        logging.info("Multiprocessing: Pool of parallel Workers is shut down.")
