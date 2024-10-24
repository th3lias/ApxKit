import numpy as np

from function.parametrized_f import ParametrizedFunction
from function.type import FunctionType
from function.utils import oscillatory, product_peak, corner_peak, gaussian, continuous, discountinuous_1d, \
    discountinuous_nd, g_function, morokoff_calfisch_1, morokoff_calfisch_2, roos_arnold, bratley, zhou


class ParametrizedFunctionProvider:
    @staticmethod
    def get_function(function_type: FunctionType, d, c, w) -> ParametrizedFunction:
        if not isinstance(function_type, FunctionType):
            raise ValueError("function_type must be of type FunctionType.")
        if c is None or w is None:
            raise ValueError("c and w must be provided.")
        match function_type:
            case FunctionType.OSCILLATORY:
                exe = lambda x: oscillatory(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Oscillatory")
            case FunctionType.PRODUCT_PEAK:
                exe = lambda x: product_peak(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Product Peak")
            case FunctionType.CORNER_PEAK:
                exe = lambda x: corner_peak(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Corner Peak")
            case FunctionType.GAUSSIAN:
                exe = lambda x: gaussian(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Gaussian")
                pass
            case FunctionType.CONTINUOUS:
                exe = lambda x: continuous(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Continuous")
            case FunctionType.DISCONTINUOUS:
                if d == 1:
                    exe = lambda x: discountinuous_1d(x, d, c, w)
                else:
                    exe = lambda x: discountinuous_nd(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Discontinuous")
            case FunctionType.G_FUNCTION:
                if c is None:
                    c = (np.arange(1, d + 1, dtype=np.float64) - 2) / 2
                if w is None:
                    w = np.zeros(d)
                exe = lambda x: g_function(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="G Function")
            case FunctionType.MOROKOFF_CALFISCH_1:
                if c is None:
                    c = np.ones(d)
                if w is None:
                    w = np.zeros(d)
                exe = lambda x: morokoff_calfisch_1(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Morokoff Calfisch 1")
            case FunctionType.MOROKOFF_CALFISCH_2:
                if c is None:
                    c = np.ones(d)
                if w is None:
                    w = np.zeros(d)
                exe = lambda x: morokoff_calfisch_2(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Morokoff Calfisch 2")
            case FunctionType.ROOS_ARNOLD:
                if c is None:
                    c = np.ones(d)
                if w is None:
                    w = np.zeros(d)
                exe = lambda x: roos_arnold(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Roos Arnold")
            case FunctionType.BRATLEY:
                if c is None:
                    c = np.ones(d)
                if w is None:
                    w = np.zeros(d)
                exe = lambda x: bratley(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Bratley")
            case FunctionType.ZHOU:
                if c is None:
                    c = 10 * np.ones(d)
                if w is None:
                    w = 1 / 3 * np.ones(d)
                exe = lambda x: zhou(x, d, c, w)
                return ParametrizedFunction(exe, d, c, w, name="Zhou")
