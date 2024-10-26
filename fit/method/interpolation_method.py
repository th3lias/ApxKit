from fit.method.fit_method import FitMethod


class InterpolationMethod(FitMethod):
    STANDARD = 1,
    LAGRANGE = 2, # TODO [Jakob]: 1 is already known method, 2 is not implemented and 3 is the new one -> adapt as soon as we know what we want
    Tasmanian = 3
