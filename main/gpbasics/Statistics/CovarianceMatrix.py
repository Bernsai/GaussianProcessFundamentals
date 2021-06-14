import gpbasics.global_parameters as global_param
global_param.ensure_init()

from enum import Enum
from typing import List

import tensorflow as tf

import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.Kernel as k
import gpbasics.KernelBasics.Operators as op
import gpbasics.KernelBasics.PartitionOperator as po


class CovarianceMatrixType(Enum):
    HOLISTIC = 0
    SEGMENTED = 1
    GLOBALIZED_SEGMENTED = 2


class CovarianceMatrix:
    def __init__(self, matrix_type: CovarianceMatrixType, kernel: k.Kernel):
        self.kernel: k.Kernel = kernel
        self.data_input: di.DataInput = None
        self.type: CovarianceMatrixType = matrix_type
        self.K: tf.Tensor = None
        self.noised_K: tf.Tensor = None
        self.K_ss: tf.Tensor = None
        self.noised_K_ss: tf.Tensor = None
        self.L_K_ss: tf.Tensor = None
        self.L_K: tf.Tensor = None
        self.L_inv_K: tf.Tensor = None
        self.K_inv: tf.Tensor = None
        self.K_s: tf.Tensor = None
        self.L_alpha: tf.Tensor = None

    def reset(self):
        """
            Method resets all obtainable matrices and sub-matrices of covariance function as well as further auxiliary
            matrices. Reseting needs to be done, if hyper parameters changed, as subsequently the resulting covariance
            matrix changed. Reseting those matrices also means, that they have to be recalculated. Thus, reseting _may_
            be an imperative, but entails further calculations.
        """
        self.K = None
        self.noised_K = None
        self.K_ss = None
        self.noised_K_ss = None
        self.L_K_ss = None
        self.L_K = None
        self.L_inv_K = None
        self.K_inv = None
        self.K_s = None
        self.L_alpha = None

    def set_data_input(self, data_input: di.AbstractDataInput):
        """
            Sets data input that is used for calculating / determining all the matrices that are obtainable from
            CovarianceMatrix. This methods also calls "reset()" as previously calculated matrices are obsolete.
        :param data_input: The newly given data_input
        """
        self.data_input: di.AbstractDataInput = data_input
        self.reset()

    def is_segmented(self) -> bool:
        """
            Indicates whether this CovarianceMatrix is a segmented one, by definition of the CovarianceMatrixType. This
            function does not check, whether the given kernel (self.kernel) allows for the usage of a segmented
            CovarianceMatrix.
        :return: boolean. TRUE self.type == CovarianceMatrixType.SEGMENTED, FALSE else
        """
        return self.type == CovarianceMatrixType.SEGMENTED

    def get_K(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        """
        Retrieves Matrix K of Covariance Matrix. K is computed using kernel-function k(x,x') for all x and x' of the
        training dataset.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of training data points according to the given data input.
        """
        pass

    def get_K_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix K_noised of Covariance Matrix. K is computed using kernel-function k(x,x') for all x and x' of the
        training dataset. K_noised further adds the predefined jitter (global_param.p_cov_matrix_jitter) on the matrix'
        diagonal.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of training data points according to the given data input.
        """
        pass

    def get_K_ss(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        """
        Retrieves Matrix K_ss of Covariance Matrix. K_ss is computed using kernel-function k(x,x') for all x and x' of
        the test dataset.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of test data points according to the given data input.
        """
        pass

    def get_K_ss_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix K_ss_noised of Covariance Matrix. K_ss is computed using kernel-function k(x,x') for all x and x' of
        the test dataset. K_ss_noised further adds the predefined jitter (global_param.p_cov_matrix_jitter) on the matrix'
        diagonal.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of test data points according to the given data input.
        """
        pass

    def get_L_K_ss(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix L_K_ss of Covariance Matrix. K_ss is computed using kernel-function k(x,x') for all x and x' of
        the test dataset. L_K_ss represents the cholesky decomposition of the noised version (K_ss_noised) of K_ss.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of test data points according to the given data input.
        """
        pass

    def get_L_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix L_K of Covariance Matrix. K is computed using kernel-function k(x,x') for all x and x' of
        the train dataset. L_K represents the cholesky decomposition of the noised version (K_noised) of K.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of train data points according to the given data input.
        """
        pass

    def get_L_inv_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix inversion of the cholesky decomposition of K of Covariance Matrix.
        K is computed using kernel-function k(x,x') for all x and x' of  the train dataset.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of train data points according to the given data input.
        """
        pass

    def get_K_inv(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves Matrix inversion of K of Covariance Matrix. K is computed using kernel-function k(x,x')
        for all x and x' of  the train dataset.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of train data points according to the given data input.
        """
        pass

    def get_K_s(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        """
        Retrieves Matrix K_s of Covariance Matrix. K_s is computed using kernel-function k(x,x')
        for all x of the training dataset and for all x' of the test dataset.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x m matrix. n: amount of train data points according to the given data input,
        m: amount of test data points according the the given data input.
        """
        pass

    def get_L_alpha(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        """
        Retrieves a special precomputed matrix especially used for log marginal likelihood calculation. This method
        retrieves the resulting Matrix of calculating: (L^T)\L^\y. Having L as the cholesky decompostion of K,
        and y as the target variable of the training data.
        :param noise:
        :param hyper_parameter:
        :return: tf.Tensor, representing a n x n matrix. n: amount of train data points according to the given data input.
        """
        pass


class HolisticCovarianceMatrix(CovarianceMatrix):
    """
    HolisticCovarianceMatrix represents the usual case of the Covariance Matrix for a Gaussian Process.
    This implementation encompasses no additional optimizations.
    """

    def __init__(self, kernel: k.Kernel):
        super(HolisticCovarianceMatrix, self).__init__(CovarianceMatrixType.HOLISTIC, kernel)

    def get_K(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K is None:
                self.K = self.kernel.get_tf_tensor(hyper_parameter, self.data_input.data_x_train,
                                                   self.data_input.data_x_train)

            return self.K

        raise Exception("No Data Input given")

    def get_K_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if noise is not None and noise.shape == []:
            if self.data_input is not None:
                if self.noised_K is None:
                    self.noised_K = tf.add(self.get_K(hyper_parameter), noise *
                                           tf.eye(self.data_input.n_train, dtype=global_param.p_dtype), name="noised_K")

                return self.noised_K

        raise Exception("No Data Input given or Noise unspecified")

    def get_K_inv(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_inv is None:
                self.K_inv = tf.linalg.inv(self.get_K_noised(hyper_parameter, noise), name="inv_K")
                # self.K_inv = tfext.subzero_cleaned_inverse(self.get_K(hyper_param), name="inv_K")

            return self.K_inv

        raise Exception("No Data Input given")

    def get_K_ss(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_ss is None:
                self.K_ss = self.kernel.get_tf_tensor(hyper_parameter, self.data_input.data_x_test,
                                                      self.data_input.data_x_test)

            return self.K_ss
        raise Exception("No Data Input given")

    def get_K_ss_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if noise is not None and noise.shape == []:
            if self.data_input is not None:
                if self.noised_K_ss is None:
                    self.noised_K_ss = tf.add(self.get_K_ss(hyper_parameter), noise *
                                              tf.eye(self.data_input.n_test, dtype=global_param.p_dtype), name="noised_K")

                return self.noised_K_ss

        raise Exception("No Data Input given or noise unspecified")

    def get_L_K_ss(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_K_ss is None:
                self.L_K_ss = tf.linalg.cholesky(self.get_K_ss_noised(hyper_parameter, noise), name="L_K_ss")

            return self.L_K_ss

        raise Exception("No Data Input given")

    def get_L_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_K is None:
                self.L_K = tf.linalg.cholesky(self.get_K_noised(hyper_parameter, noise), name="L_K")

            return self.L_K

        raise Exception("No Data Input given")

    def get_L_alpha(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_alpha is None:
                L = self.get_L_K(hyper_parameter, noise)
                self.L_alpha = tf.linalg.triangular_solve(
                    tf.linalg.matrix_transpose(L),
                    tf.linalg.triangular_solve(L, self.data_input.get_detrended_y_train()), lower=False)
            return self.L_alpha

        raise Exception("No Data Input given")

    def get_L_inv_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_inv_K is None:
                self.L_inv_K = tf.linalg.inv(self.get_L_K(hyper_parameter, noise), name="inv_L_K")
                # self.L_inv_K = tfext.subzero_cleaned_inverse(self.get_L_K(hyper_param), name="inv_L_K")

            return self.L_inv_K

        raise Exception("No Data Input given")

    def get_K_s(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_s is None:
                with tf.name_scope("K_s"):
                    self.K_s = self.kernel.get_tf_tensor(hyper_parameter, self.data_input.data_x_train,
                                                         self.data_input.data_x_test)

            return self.K_s

        raise Exception("No Data Input given")


class SegmentedCovarianceMatrix(CovarianceMatrix):
    """
        SegmentedCovarianceMatrix represents the a special case of the Covariance Matrix for a Gaussian Process.
        It relies on a composite covariance function that uses a change point operator at the top level, to divide the
        given dataset up into independent local models. It enables to exploit the resulting block matrix shape of the
        covariance matrix. This results in a significant speed-up of certain calculations.
    """

    def __init__(self, kernel: po.PartitionOperator):
        super(SegmentedCovarianceMatrix, self).__init__(CovarianceMatrixType.SEGMENTED, kernel)

    def set_data_input(self, data_input: di.PartitionedDataInput):
        assert len(data_input.data_inputs) == len(self.kernel.child_nodes), \
            "Invalid data input. Data input does not match segments prescribed by given kernel"
        super(SegmentedCovarianceMatrix, self).set_data_input(data_input)

    def get_K(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K is None:
                list_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_blocks(hyper_parameter)

                self.K = tf.linalg.LinearOperatorBlockDiag([block_k for block_k in list_K if block_k is not None])

            return self.K.to_dense()

        raise Exception("No Data Input given")

    def get_K_blocks(self, hyper_parameter) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_K: List[tf.linalg.LinearOperatorFullMatrix] = []
        hyper_parameter_index = 0
        if isinstance(self.kernel, op.ChangePointOperator):
            hyper_parameter_index = len(self.kernel.change_point_positions)
        for i in range(0, len(self.kernel.child_nodes)):
            block_input: di.DataInput = self.data_input.data_inputs[i]

            child_node = self.kernel.child_nodes[i]
            number_hyper_parameter = child_node.get_number_of_hyper_parameter()
            if block_input.n_train > 0:
                sliced_hyper_parameter: List[tf.Tensor] = \
                    hyper_parameter[hyper_parameter_index:
                                    hyper_parameter_index + number_hyper_parameter]

                K = child_node.get_tf_tensor(
                    sliced_hyper_parameter, block_input.data_x_train, block_input.data_x_train)
                list_K.append(tf.linalg.LinearOperatorFullMatrix(K))
            else:
                list_K.append(None)

            hyper_parameter_index += number_hyper_parameter

        return list_K

    def get_K_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.noised_K is None:
                list_noised_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_noised_blocks(hyper_parameter, noise)

                self.noised_K = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_noised_K if block_k is not None])

            return self.noised_K.to_dense()

        raise Exception("No Data Input given")

    def get_K_noised_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        if noise is not None and noise.shape == []:
            list_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_blocks(hyper_parameter)
            list_noised_K: List[tf.linalg.LinearOperatorFullMatrix] = []
            for i in range(0, len(self.kernel.child_nodes)):
                block_input = self.data_input.data_inputs[i]
                if block_input.n_train > 0:
                    In = noise * tf.eye(block_input.n_train, dtype=global_param.p_dtype)
                    block_noised_K = list_K[i].add_to_tensor(In)
                    list_noised_K.append(tf.linalg.LinearOperatorFullMatrix(block_noised_K))
                else:
                    list_noised_K.append(None)
            return list_noised_K
        raise Exception("Invalid Noise given")

    def get_K_ss(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_ss is None:
                list_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_ss_blocks(hyper_parameter)

                self.K_ss = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_K_ss if block_k is not None])

            return self.K_ss.to_dense()

        raise Exception("No Data Input given")

    def get_K_ss_blocks(self, hyper_parameter) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = []
        hyper_parameter_index = 0
        if isinstance(self.kernel, op.ChangePointOperator):
            hyper_parameter_index = len(self.kernel.change_point_positions)
        for i in range(0, len(self.kernel.child_nodes)):
            block_input: di.DataInput = self.data_input.data_inputs[i]

            child_node = self.kernel.child_nodes[i]
            number_hyper_parameter = child_node.get_number_of_hyper_parameter()
            if block_input.n_test > 0:
                sliced_hyper_parameter: List[tf.Tensor] = \
                    hyper_parameter[hyper_parameter_index:
                                    hyper_parameter_index + number_hyper_parameter]

                K_ss = child_node.get_tf_tensor(
                    sliced_hyper_parameter, block_input.data_x_test, block_input.data_x_test)
                list_K_ss.append(tf.linalg.LinearOperatorFullMatrix(K_ss))
            else:
                list_K_ss.append(None)

            hyper_parameter_index += number_hyper_parameter

        return list_K_ss

    def get_K_ss_noised(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.noised_K_ss is None:
                list_noised_K_ss = self.get_K_ss_noised_blocks(hyper_parameter, noise)

                self.noised_K_ss = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_noised_K_ss if block_k is not None])

            return self.noised_K_ss.to_dense()

        raise Exception("No Data Input given")

    def get_K_ss_noised_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        if noise is not None and noise.shape == []:
            list_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_ss_blocks(hyper_parameter)
            list_noised_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = []
            for i in range(0, len(self.kernel.child_nodes)):
                block_input = self.data_input.data_inputs[i]
                if block_input.n_test > 0:
                    Is = noise * tf.eye(block_input.n_test, dtype=global_param.p_dtype)
                    block_noised_K_ss = list_K_ss[i].to_dense() + Is
                    list_noised_K_ss.append(tf.linalg.LinearOperatorFullMatrix(block_noised_K_ss))
                else:
                    list_noised_K_ss.append(None)
            return list_noised_K_ss
        raise Exception("Invalid noise provided")

    def get_L_K_ss(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_K_ss is None:
                list_L_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = self.get_L_K_ss_blocks(hyper_parameter, noise)

                self.L_K_ss = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_L_K_ss if block_k is not None])

            return self.L_K_ss.to_dense()

        raise Exception("No Data Input given")

    def get_L_K_ss_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_noised_K_ss: List[tf.linalg.LinearOperatorFullMatrix] = \
            self.get_K_ss_noised_blocks(hyper_parameter, noise)
        list_L_K_ss = []
        for i in range(0, len(self.kernel.child_nodes)):
            k_ss_i = list_noised_K_ss[i]
            if k_ss_i is not None:
                block_L_K_ss = tf.linalg.cholesky(k_ss_i.to_dense())
                list_L_K_ss.append(tf.linalg.LinearOperatorFullMatrix(block_L_K_ss))
            else:
                list_L_K_ss.append(None)
        return list_L_K_ss

    def get_L_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_K is None:
                list_L_K = self.get_L_K_blocks(hyper_parameter, noise)

                self.L_K = tf.linalg.LinearOperatorBlockDiag(
                    [tf.linalg.LinearOperatorFullMatrix(block_k) for block_k in list_L_K if block_k is not None])

            return self.L_K.to_dense()

        raise Exception("No Data Input given")

    def get_L_K_blocks(self, hyper_parameter, noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_noised_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_noised_blocks(hyper_parameter, noise)
        list_L_K = []
        for i in range(0, len(self.kernel.child_nodes)):
            k_i = list_noised_K[i]
            if k_i is not None:
                block_L_K = tf.linalg.cholesky(k_i.to_dense())
                list_L_K.append(block_L_K)
            else:
                list_L_K.append(None)
        return list_L_K

    def get_L_alpha(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_alpha is None:
                list_L_alpha: List[tf.linalg.LinearOperatorFullMatrix] = self.get_L_alpha_blocks(hyper_parameter, noise)

                self.L_alpha = tf.concat([block_k for block_k in list_L_alpha if block_k is not None], axis=0)

            return self.L_alpha

        raise Exception("No Data Input given")

    def get_L_alpha_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_L_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_L_K_blocks(hyper_parameter, noise)
        list_L_alpha: List[tf.linalg.LinearOperatorFullMatrix] = []
        for i in range(0, len(self.kernel.child_nodes)):
            L = list_L_K[i]
            if L is not None:
                # lower parameters are crucial!
                block_L_alpha = tf.linalg.triangular_solve(
                    tf.transpose(L), tf.linalg.triangular_solve(
                        L, self.data_input.data_inputs[i].get_detrended_y_train(), lower=True), lower=False)

                list_L_alpha.append(block_L_alpha)
            else:
                list_L_alpha.append(None)
        return list_L_alpha

    def get_L_inv_K(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.L_inv_K is None:
                list_L_inv_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_L_inv_K_blocks(hyper_parameter, noise)

                self.L_inv_K = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_L_inv_K if block_k is not None])

            return self.L_inv_K.to_dense()

        raise Exception("No Data Input given")

    def get_L_inv_K_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_L_K: List[tf.linalg.LinearOperatorFullMatrix] = self.get_L_K_blocks(hyper_parameter, noise)
        list_L_inv_K: List[tf.linalg.LinearOperatorFullMatrix] = []
        for i in range(0, len(self.kernel.child_nodes)):
            k_i = list_L_K[i]
            if k_i is not None:
                block_K_inv = tf.linalg.inv(k_i)
                list_L_inv_K.append(tf.linalg.LinearOperatorFullMatrix(block_K_inv))
            else:
                list_L_inv_K.append(None)
        return list_L_inv_K

    def get_K_inv(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_inv is None:
                list_K_inv: List[tf.linalg.LinearOperatorFullMatrix] = self.get_K_inv_blocks(hyper_parameter, noise)

                self.K_inv = tf.linalg.LinearOperatorBlockDiag(
                    [block_k for block_k in list_K_inv if block_k is not None])

            return self.K_inv.to_dense()

        raise Exception("No Data Input given")

    def get_K_inv_blocks(self, hyper_parameter: List[tf.Tensor], noise: tf.Tensor) -> List[tf.linalg.LinearOperatorFullMatrix]:
        list_noised_K = self.get_K_noised_blocks(hyper_parameter, noise)
        list_K_inv = []
        for i in range(0, len(self.kernel.child_nodes)):
            k_i = list_noised_K[i]
            if k_i is not None:
                block_K_inv = tf.linalg.inv(k_i.to_dense())
                list_K_inv.append(tf.linalg.LinearOperatorFullMatrix(block_K_inv))
            else:
                list_K_inv.append(None)
        return list_K_inv

    def get_K_s(self, hyper_parameter: List[tf.Tensor]) -> tf.Tensor:
        if self.data_input is not None:
            if self.K_s is None:
                with tf.name_scope("K_s"):
                    self.K_s = self.kernel.get_tf_tensor(hyper_parameter, self.data_input.data_x_train,
                                                         self.data_input.data_x_test)

            return self.K_s

        raise Exception("No Data Input given")
