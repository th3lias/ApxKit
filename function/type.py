from enum import Enum


class FunctionType(Enum):
    BNR_OSCILLATORY = 1,
    BNR_PRODUCT_PEAK = 2,
    BNR_CORNER_PEAK = 3,
    BNR_GAUSSIAN = 4,
    BNR_CONTINUOUS = 5,
    BNR_DISCONTINUOUS = 6,
    G_FUNCTION = 7,
    MOROKOFF_CALFISCH_1 = 8,
    MOROKOFF_CALFISCH_2 = 9,
    ROOS_ARNOLD = 10,
    BRATLEY = 11,
    ZHOU = 12,
    NOISE = 13,
    