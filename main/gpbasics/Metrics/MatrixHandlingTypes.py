from enum import Enum


class GlobalApproximationsType(Enum):
    pass


class MatrixApproximations(GlobalApproximationsType):
    NONE = 0
    SKC_LOWER_BOUND = 1
    SKC_UPPER_BOUND = 2

    # Basic Nystroem does not involve optimization by means of trace(Covariance Matrix)
    BASIC_NYSTROEM = 3

    # Structured Kernel Interpolation by Wilson, Nickisch ICML 2015
    SKI = 4


class SubsetOfDataApproaches(GlobalApproximationsType):
    # Subset-Of-Data approaches
    ## Taking random points from input data set x and accompanying target values y
    SOD_RANDOM = 5

    ## Taking grid like input data x and accompanying target values y
    SOD_GRID = 6

    ## Taking true grid regarding the domain of input data x and find approximate target values by kernel-smoothing
    SOD_SMOOTHED_GRID = 7


class NumericalMatrixHandlingType(Enum):
    STRICT_INVERSE = 0
    PSEUDO_INVERSE = 1
    CHOLESKY_BASED = 2
    LINEAR_CONJUGATE_GRADIENT = 3
