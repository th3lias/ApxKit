from typing import Union, Callable, List

import Tasmanian
import numpy as np

from interpolate.interpolator import Interpolator


class TasmanianInterpolator(Interpolator):
    def interpolate(self, f: Union[Callable, List[Callable]]) -> np.ndarray:
        pass
