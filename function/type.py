from enum import Enum


class FunctionType(Enum):
    OSCILLATORY = 1,
    PRODUCT_PEAK = 2,
    CORNER_PEAK = 3,
    GAUSSIAN = 4,
    CONTINUOUS = 5,
    DISCONTINUOUS = 6,
    G_FUNCTION = 7,
    MOROKOFF_CALFISCH_1 = 8,
    MOROKOFF_CALFISCH_2 = 9,
    ROOS_ARNOLD = 10,
    BRATLEY = 11,
    ZHOU = 12,
    NOISE = 13,
    