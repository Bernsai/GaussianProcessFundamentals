from enum import Enum


class GlobalApproximations(Enum):
    Nystroem = 0,
    SKC = 1,
    SKI = 2,

    # SOD = Subset of data
    SOD_Grid = 3,
    SOD_Random = 4,
    SOD_SmoothedGrid = 5
