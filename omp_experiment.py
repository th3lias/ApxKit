"""
Small trial run: OMP vs LS vs WLS on a subset of Genz functions.
Dimensions 2–4, scales 1–8, three smooth function types, 10 instances each.
"""
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import torch

from algorithm.least_squares import LeastSquaresAlgorithm
from algorithm.omp import OMP
from algorithm.weighted_least_squares import WeightedLeastSquaresAlgorithm
from basis.clenshaw_curtis_level_polynomial_basis_generator import ClenshawCurtisLevelPolynomialBasisGenerator
from experiment.experiment_executor import ExperimentExecutor
from function.type import FunctionType
from grid.generator.chebyshev_grid_generator import ChebyshevGridGenerator
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.generator.uniform_number_generator import UniformNumberGenerator
from main import _plot_errors
from solver.scipy_lstsq_solver import ScipyLstsqSolver

# ── Parameters ────────────────────────────────────────────────────────────────
dim_scale_dict = {
        2: [1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        3: [1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        4: [1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        5: [1, 2, 3, 4, 5, 6, 7, 8, ],
        6: [1, 2, 3, 4, 5, 6, 7, ],
        7: [1, 2, 3, 4, 5, 6, 7, ],
        8: [1, 2, 3, 4, 5, 6, ],
        9: [1, 2, 3, 4, 5, 6, ],
        10: [1, 2, 3, 4, 5, 6, ],
    }
function_types = [FunctionType.ZHOU, FunctionType.CONTINUOUS, FunctionType.CORNER_PEAK,
                      FunctionType.DISCONTINUOUS, FunctionType.GAUSSIAN, FunctionType.MOROKOFF_CALFISCH_1,
                      FunctionType.G_FUNCTION, FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.NOISE]
n_fun_parallel  = 3
avg_c           = {ft: 1.0 for ft in function_types}
seed_list       = [42, 44, 45, 46]
multiplier      = lambda x: 2 * x

# ── Components ────────────────────────────────────────────────────────────────
basis  = ClenshawCurtisLevelPolynomialBasisGenerator(store_indices=True)
solver = ScipyLstsqSolver(driver='gelsy')
device = (torch.device("mps")  if torch.backends.mps.is_available()  else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

uniform_gen = UniformGridGenerator(seed=42, multiplier_fun=multiplier)
cheb_gen    = ChebyshevGridGenerator(seed=43, multiplier_fun=multiplier)
omp_gen     = ChebyshevGridGenerator(seed=47, multiplier_fun=multiplier)
test_gen    = UniformGridGenerator(seed=44)

c_gen = UniformNumberGenerator(seed=45)
w_gen = UniformNumberGenerator(seed=46)

# ── Algorithms ────────────────────────────────────────────────────────────────
ls  = LeastSquaresAlgorithm(basis, uniform_gen, solver)
wls = WeightedLeastSquaresAlgorithm(basis, cheb_gen, solver)
omp = OMP(grid_generator=omp_gen, num_iters=20_000, tol=1e-6)

# ── Run ───────────────────────────────────────────────────────────────────────
ex = ExperimentExecutor(
    dim_scale_dict,
    test_grid_generator=test_gen,
    uniform_value_generator_c=c_gen,
    uniform_value_generator_w=w_gen,
    use_max_scale=False,
    path="results/omp_trial/results.csv",
)

df = ex.execute_experiments(
    [ls, wls, omp], function_types, n_fun_parallel,
    avg_c=avg_c, seed_list=seed_list, device=device,
    plot_callbacks=[_plot_errors],
)

# ── Summary ───────────────────────────────────────────────────────────────────
summary = (df.groupby(['algorithm', 'dim', 'scale'])['ell_2_error']
             .median()
             .reset_index()
             .rename(columns={'ell_2_error': 'median_ell2'}))
summary['median_ell2'] = summary['median_ell2'].map('{:.2e}'.format)
print('\nMedian L2 error by algorithm / dim / scale')
print(summary.to_string(index=False))
