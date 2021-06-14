import gpbasics.global_parameters as global_param
global_param.ensure_init()

from enum import Enum
from typing import List
import logging
import numpy as np
import gpbasics.DataHandling.DataInput as di
import tensorflow as tf


class PartitioningClass(Enum):
    # score is 1 for one partition and zero for all the others.
    # If one partition has score=1 no other partition needs to be checked.
    SELF_SUFFICIENT = 0,

    # score may be any float. data record belongs to that partition with the smallest distance
    SMALLEST_DISTANCE = 1


class PartitionCriterion:
    def __init__(self, partitioning_type: PartitioningClass):
        self.partitioning_type: PartitioningClass = partitioning_type

    def get_score(self, x_vector: np.ndarray) -> np.ndarray:
        pass

    def deepcopy(self):
        pass

    def get_json(self) -> dict:
        pass


class PartitioningModel:
    def __init__(self, partition_class: PartitioningClass, ignored_dimensions: List[int]):
        self.partitioning: List[PartitionCriterion] = []
        self.partition_class: PartitioningClass = partition_class
        self.ignored_dimensions: List[int] = ignored_dimensions

    def automatic_init_criteria(self, data_input: di.DataInput, optimize_metric,
                                model_selection_metric, number_of_partitions: int = None,
                                predecessor_criterion: PartitionCriterion = None):
        pass

    def init_partitioning(self, partitioning: List[PartitionCriterion]):
        if len(self.partitioning) > 0:
            logging.warning("%s: Overwriting old partitioning." % str(self))
        self.partitioning = partitioning

    def get_number_of_partitions(self) -> int:
        return len(self.partitioning)

    def add_partitioning_criterion(self, criterion: PartitionCriterion):
        assert criterion.partitioning_type == self.partition_class, \
            "Partitioning Criterion does not match Partitioning Model"

        assert criterion is not None, "Criterion cannot be None"

        self.partitioning.append(criterion)

    def partition_data_input(self, data_input: di.DataInput) -> di.PartitionedDataInput:
        if len(self.partitioning) <= 1:
            logging.warning("Dataset cannot be partitioned as only one / none partition criterion is available.")
            return di.PartitionedDataInput(data_input.data_x_train, data_input.data_y_train, data_input.data_x_test,
                                           data_input.data_y_test, [data_input])

        separate_test_train: bool = not np.array_equal(data_input.data_x_train, data_input.data_x_test)

        data_inputs: List[di.DataInput] = []

        train_indices: List[np.ndarray] = self.get_data_record_indices_per_partition(data_input.data_x_train)
        test_indices: List[np.ndarray]

        if separate_test_train:
            test_indices = self.get_data_record_indices_per_partition(data_input.data_x_test)
        else:
            test_indices = train_indices

        assert len(train_indices) == len(test_indices)

        x_train, y_train, x_test, y_test = None, None, None, None

        for i in range(len(train_indices)):
            block_x_train = tf.gather(data_input.data_x_train, train_indices[i])
            block_x_test = tf.gather(data_input.data_x_test, test_indices[i])
            block_y_train = tf.gather(data_input.data_y_train, train_indices[i])
            if data_input.data_y_test is not None:
                block_y_test = tf.gather(data_input.data_y_test, test_indices[i])
            else:
                block_y_test = None
            block_data_input: di.DataInput = di.DataInput(data_x_train=block_x_train, data_x_test=block_x_test,
                                                          data_y_train=block_y_train, data_y_test=block_y_test)
            data_inputs.append(block_data_input)

            if x_train is None:
                x_train, y_train, x_test, y_test = block_x_train, block_y_train, block_x_test, block_y_test
            else:
                x_train = tf.concat([x_train, block_x_train], axis=0)
                y_train = tf.concat([y_train, block_y_train], axis=0)
                x_test = tf.concat([x_test, block_x_test], axis=0)
                if block_y_test is not None and y_test is not None:
                    y_test = tf.concat([y_test, block_y_test], axis=0)
                else:
                    y_test = None

        return di.PartitionedDataInput(x_train, y_train, x_test, y_test, data_inputs)

    def get_data_record_indices_per_partition(self, x_vector: np.ndarray) -> List[np.ndarray]:
        score_columns: List[np.ndarray] = []

        for criterion in self.partitioning:
            score_columns.append(criterion.get_score(self.filter_data_by_ignored_dimensions(x_vector)))

        score_matrix_train: np.ndarray = np.transpose(np.array(score_columns))

        if self.partition_class == PartitioningClass.SMALLEST_DISTANCE:
            score_matrix_train = score_matrix_train + np.random.normal(0, 1e-10, score_matrix_train.shape)
            col_min: np.ndarray = np.amin(score_matrix_train, axis=1)
            score_matrix_train = score_matrix_train == col_min.reshape(-1, 1)

        indices_per_partition: List[np.ndarray] = []

        for i in range(self.get_number_of_partitions()):
            indices = np.where(score_matrix_train[:, i] == 1)[0]
            indices_per_partition.append(indices)

        if len(indices_per_partition) == 0:
            indices_per_partition = [np.linspace(0, len(x_vector) - 1, len(x_vector), dtype=int)]

        return indices_per_partition

    def filter_data_by_ignored_dimensions(self, vector: np.ndarray):
        if len(self.ignored_dimensions) == 0:
            return vector

        assert vector.shape[1] > max(self.ignored_dimensions)

        bool_mask = [not i in self.ignored_dimensions for i in range(vector.shape[1])]

        return vector[:, bool_mask]

    def deepcopy(self):
        partitioning: List[PartitionCriterion] = [pc.deepcopy() for pc in self.partitioning]
        return PartitioningModel(partitioning, self.partition_class)

    def get_hash_tuple(self):
        return tuple(self.ignored_dimensions) + (sum([hash(crit) for crit in self.partitioning]), )

    def __hash__(self):
        return hash(self.get_hash_tuple())
