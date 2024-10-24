from fit.method.fit_method import FitMethod


class LeastSquaresMethod(FitMethod):
    EXACT = 1,
    NUMPY_LSTSQ = 2
