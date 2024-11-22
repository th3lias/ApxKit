from fit.method.fit_method import FitMethod


class LeastSquaresMethod(FitMethod):
    EXACT = 1,
    SCIPY_LSTSQ_GELSD = 2
    SCIPY_LSTSQ_GELSS = 3
    SCIPY_LSTSQ_GELSY = 4

