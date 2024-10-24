#  Created 2024. (Elias Mindlberger)

#  Created 2024. (Elias Mindlberger)

#  Created 2024. (Elias Mindlberger)
from fit.method.fit_method import FitMethod


class LeastSquaresMethod(FitMethod):
    EXACT = 1,
    ITERATIVE_LSMR = 2,
    SKLEARN = 3,
    PYTORCH = 4,
    PYTORCH_NEURAL_NET = 5,
    RLS = 7,
    ITERATIVE_RLS = 8,
    NUMPY_LSTSQ = 9
