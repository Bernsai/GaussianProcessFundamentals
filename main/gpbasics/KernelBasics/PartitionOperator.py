import gpbasics.global_parameters as global_param
global_param.ensure_init()

from gpbasics.Auxiliary.NonSquareBlockMatrices import build_block_matrix_from_non_square_matrices

from typing import List
import tensorflow as tf
import numpy as np

import gpbasics.KernelBasics.Operators as op
import gpbasics.KernelBasics.Kernel as k
import gpbasics.KernelBasics.PartitioningModel as pm


class PartitionOperator(op.Operator):
    def __init__(self, input_dimensionality: int, child_nodes: List[k.Kernel],
                 partitioning_model: pm.PartitioningModel):
        assert len(child_nodes) == partitioning_model.get_number_of_partitions(), \
            "One partitioning criterion for each kernel has to be supplied"
        super(PartitionOperator, self).__init__(k.KernelManifestation.PART, input_dimensionality, child_nodes)
        self.operator_sign: str = "|"
        self.partitioning_model: pm.PartitioningModel = partitioning_model

    def get_tf_tensor(self, hyper_parameter: List[tf.Tensor], x_vector: np.ndarray, x_vector_: np.ndarray) -> tf.Tensor:
        block_matrices, square_block_matrices, _, _ = \
            self.get_list_of_block_matrices(hyper_parameter, x_vector, x_vector_)

        result: tf.Tensor

        if square_block_matrices:
            result = tf.linalg.LinearOperatorBlockDiag(
                [tf.linalg.LinearOperatorFullMatrix(bm)
                 for bm in filter(lambda block: isinstance(block, tf.Tensor), block_matrices)]).to_dense()
        else:
            block_matrix = build_block_matrix_from_non_square_matrices(block_matrices)

            result = block_matrix

        # Shape of covariance matrix need to fit
        # x_vector_.shape[1] and x_vector.shape[1] are ignored as these represent the datasets dimensionality,
        # which is not reflected in the shape of the covariance matrix, which is always shaped like:
        # shape=[x_vector.shape[0], x_vector_.shape[0]]
        result_shape = tf.shape(result)
        assert result_shape[0] == x_vector.shape[0] and result_shape[1] == x_vector_.shape[0], \
            "x_vector.shape=%s, result_shape=%s" % (str(x_vector.shape), str(result_shape))

        return result

    def get_list_of_block_matrices(self, hyper_parameter, x_vector, x_vector_):
        indices: List[np.ndarray] = self.partitioning_model.get_data_record_indices_per_partition(x_vector)
        indices_: List[np.ndarray]
        if not x_vector is x_vector_:
            indices_ = self.partitioning_model.get_data_record_indices_per_partition(x_vector_)
        else:
            indices_ = indices
        assert len(indices) == len(indices_) and len(indices) == len(self.child_nodes)
        square_block_matrices = True
        block_matrices: List[tf.Tensor] = []
        hyper_parameter_index = 0
        for i in range(len(indices)):
            child_node = self.child_nodes[i]

            number_hyper_parameter = child_node.get_number_of_hyper_parameter()

            if len(indices[i]) == 0 or len(indices_[i]) == 0:
                if len(indices[i]) != len(indices_[i]):
                    block_matrices.append([len(indices[i]), len(indices_[i])])
            else:
                child_node = self.child_nodes[i]

                number_hyper_parameter = child_node.get_number_of_hyper_parameter()
                sliced_hyper_parameter: List[tf.Tensor] = \
                    hyper_parameter[hyper_parameter_index:
                                    hyper_parameter_index + number_hyper_parameter]

                block_matrices.append(child_node.get_tf_tensor(
                    sliced_hyper_parameter, tf.gather(x_vector, indices[i]), tf.gather(x_vector_, indices_[i])))

            if len(indices[i]) != len(indices_[i]):
                square_block_matrices = False

            hyper_parameter_index += number_hyper_parameter
        return block_matrices, square_block_matrices, indices, indices_

    def add_kernel(self, kernel: k.Kernel, criterion: pm.PartitionCriterion):
        assert kernel is not None, "Adding None as kernel to ChangePoint is not allowed."

        self.child_nodes.append(kernel)
        self.partitioning_model.add_partitioning_criterion(criterion)

    def deepcopy(self):
        copied_kernel = PartitionOperator(
            self.input_dimensionality, [cn.deepcopy() for cn in self.child_nodes], self.partitioning_model.deepcopy())

        if self.noise is not None:
            copied_kernel.set_noise(self.noise)

        return copied_kernel

    def get_json(self) -> dict:
        if len(self.child_nodes) == 1:
            return {"type": self.manifestation.name, "child_nodes": [self.child_nodes[0].get_json()]}

        child_nodes = []
        for i in range(len(self.child_nodes)):
            cn_json = self.child_nodes[i].get_json()

            cn_json['partitioning_criterion'] = self.partitioning_model.partitioning[i].get_json()

            child_nodes.append(cn_json)

        result = {"type": self.manifestation.name, "child_nodes": child_nodes}

        return result

    def get_simplified_version(self):
        simplified_child_nodes = [cn.get_simplified_version() for cn in self.child_nodes]

        return PartitionOperator(self.input_dimensionality, simplified_child_nodes, self.partitioning_model)

    def get_hash_tuple(self):
        return super(PartitionOperator, self).get_hash_tuple() + tuple([hash(cn) for cn in self.child_nodes]) + \
               (hash(self.partitioning_model), )
