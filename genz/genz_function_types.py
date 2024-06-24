from enum import Enum


class GenzFunctionType(Enum):
    OSCILLATORY = 1,
    PRODUCT_PEAK = 2,
    CORNER_PEAK = 3,
    GAUSSIAN = 4,
    CONTINUOUS = 5,
    DISCONTINUOUS = 6,

