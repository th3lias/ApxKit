from fit.method.fit_method import FitMethod


class LeastSquaresMethod(FitMethod):
    EXACT = 1,
    NUMPY_LSTSQ = 2 # TODO: Make this as Scipy LSTSQ GELSD
    SCIPY_LSTSQ_GELSD = 3
    SCIPY_LSTSQ_GELSS = 4
    SCIPY_LSTSQ_GELSY = 5

