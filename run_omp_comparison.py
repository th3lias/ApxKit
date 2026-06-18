"""
OMP vs LS vs WLS comparison experiment.
See docs/omp_comparison.md for the full experimental design.

Feasible scales per dimension (N × M_OMP design matrix < ~6 GB):
    D=2: scale 1–8   (max  0.04 GB)
    D=3: scale 1–8   (max  0.59 GB)
    D=4: scale 1–8   (max  5.81 GB)
    D=5: scale 1–7   (max  6.07 GB)
    D=6: scale 1–6   (max  4.16 GB)

Usage:
    python run_omp_comparison.py                  # default config below
    python run_omp_comparison.py -f my_run        # save to results/my_run/
"""

import sys
sys.path.insert(0, '.')

import argparse
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
from plot.plot_error_distribution import plot_all_errors_fixed_dim
from solver.scipy_lstsq_solver import ScipyLstsqSolver


def _plot_errors(results_df, save_dir, d=None, s=None, verbose=False):
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d,
                              save=True, latex=True, only_maximum=False, verbose=verbose)
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d,
                              save=True, latex=True, only_maximum=True, verbose=verbose)


def run(folder_name: str = "omp_comparison"):

    # ── Scales to include per dimension ───────────────────────────────────────
    # Edit this dict to extend or reduce the experiment.
    # Excluded (memory > 6 GB): D=5 scale 8, D=6 scale 7+.
    dim_scale_dict = {
        2: [1, 2, 3, 4, 5, 6, 7, 8],
        3: [1, 2, 3, 4, 5, 6, 7, 8],
        4: [1, 2, 3, 4, 5, 6, 7, 8],
        5: [1, 2, 3, 4, 5, 6, 7],
        6: [1, 2, 3, 4, 5, 6],
    }

    # ── Function types ────────────────────────────────────────────────────────
    function_types = [
        FunctionType.CONTINUOUS,
        FunctionType.PRODUCT_PEAK,
        FunctionType.GAUSSIAN,
    ]

    # ── Reproducibility ───────────────────────────────────────────────────────
    # seed_list = [uniform_seed, test_seed, c_seed, w_seed]
    seed_list       = [42, 44, 45, 46]
    n_fun_parallel  = 10          # random function instances per type
    avg_c           = {ft: 1.0 for ft in function_types}
    multiplier      = lambda x: 2 * x   # 2× oversampling for LS/WLS/OMP grids

    # ── Grid generators ───────────────────────────────────────────────────────
    uniform_gen = UniformGridGenerator(seed=42, multiplier_fun=multiplier)
    cheb_gen    = ChebyshevGridGenerator(seed=43, multiplier_fun=multiplier)
    omp_gen     = ChebyshevGridGenerator(seed=47, multiplier_fun=multiplier)
    test_gen    = UniformGridGenerator(seed=44)

    c_gen = UniformNumberGenerator(seed=45)
    w_gen = UniformNumberGenerator(seed=46)

    # ── Basis / solver ────────────────────────────────────────────────────────
    basis  = ClenshawCurtisLevelPolynomialBasisGenerator(store_indices=True)
    solver = ScipyLstsqSolver(driver='gelsy')

    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))

    # ── Algorithms ────────────────────────────────────────────────────────────
    # hc_bandwidth=None  →  auto-selected at fit time so M_OMP ≥ M_LS
    # num_iters=20_000   →  matches SparseRecovery; early-stopped by tol=1e-4
    ls  = LeastSquaresAlgorithm(basis, uniform_gen, solver)
    wls = WeightedLeastSquaresAlgorithm(basis, cheb_gen, solver)
    omp = OMP(grid_generator=omp_gen, num_iters=20_000, tol=1e-4)

    # ── Run ───────────────────────────────────────────────────────────────────
    import os
    path = os.path.join("results", folder_name, "results.csv")

    ex = ExperimentExecutor(
        dim_scale_dict,
        test_grid_generator=test_gen,
        uniform_value_generator_c=c_gen,
        uniform_value_generator_w=w_gen,
        use_max_scale=False,
        path=path,
    )

    df = ex.execute_experiments(
        [ls, wls, omp], function_types, n_fun_parallel,
        avg_c=avg_c, seed_list=seed_list, device=device,
        plot_callbacks=[_plot_errors],
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = (df.groupby(['algorithm', 'dim', 'scale'])['ell_2_error']
                 .median()
                 .reset_index()
                 .rename(columns={'ell_2_error': 'median_ell2'}))
    summary['median_ell2'] = summary['median_ell2'].map('{:.2e}'.format)
    print('\nMedian L2 error by algorithm / dim / scale')
    print(summary.to_string(index=False))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_name', default='omp_comparison', type=str)
    args = parser.parse_args()
    run(folder_name=args.folder_name)
