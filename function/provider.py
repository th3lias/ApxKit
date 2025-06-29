import numpy as np
from typing import Union

from function.parametrized_f import ParametrizedFunction
from function.type import FunctionType
from function.utils import oscillatory, product_peak, corner_peak, gaussian, continuous, discontinuous, g_function, \
    morokoff_calfisch_1, morokoff_calfisch_2, roos_arnold, bratley, zhou, noise, zero


class ParametrizedFunctionProvider:
    @staticmethod
    def get_function(function_type: FunctionType, d: int, c: Union[np.array, None] = None,
                     w: Union[np.array, None] = None, test: bool = False) -> ParametrizedFunction:
        if not isinstance(function_type, FunctionType):
            raise ValueError("function_type must be of type FunctionType.")
        match function_type:
            case FunctionType.BNR_OSCILLATORY:
                exe = lambda x: oscillatory(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Oscillatory")
            case FunctionType.BNR_PRODUCT_PEAK:
                exe = lambda x: product_peak(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Product Peak")
            case FunctionType.BNR_CORNER_PEAK:
                exe = lambda x: corner_peak(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Corner Peak")
            case FunctionType.BNR_GAUSSIAN:
                exe = lambda x: gaussian(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Gaussian")
            case FunctionType.BNR_CONTINUOUS:
                exe = lambda x: continuous(x,d,c,w)
                return ParametrizedFunction(exe, d, c, w, name="Continuous")
            case FunctionType.BNR_DISCONTINUOUS:
                exe = lambda x: discontinuous(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Discontinuous")
            case FunctionType.G_FUNCTION:
                exe = lambda x: g_function(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="G Function")
            case FunctionType.MOROKOFF_CALFISCH_1:
                exe = lambda x: morokoff_calfisch_1(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Morokoff Calfisch 1")
            case FunctionType.MOROKOFF_CALFISCH_2:
                exe = lambda x: morokoff_calfisch_2(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Morokoff Calfisch 2")
            case FunctionType.ROOS_ARNOLD:
                exe = lambda x: roos_arnold(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Roos Arnold")
            case FunctionType.BRATLEY:
                exe = lambda x: bratley(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Bratley")
            case FunctionType.ZHOU:
                exe = lambda x: zhou(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Zhou")
            case FunctionType.NOISE:
                if test:
                    exe = lambda x: zero(x, d, c, w)
                else:
                    exe = lambda x: noise(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Noise")
            case _:
                raise ValueError(f"Unknown Function type {function_type}")
