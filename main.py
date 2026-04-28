import argparse
import os

from algorithm.least_squares import LeastSquaresAlgorithm
from algorithm.smolyak import SmolyakAlgorithm
from algorithm.weighted_least_squares import WeightedLeastSquaresAlgorithm
from basis.basis_generator import BasisGenerator
from basis.clenshaw_curtis_level_polynomial_basis_generator import ClenshawCurtisLevelPolynomialBasisGenerator
from basis.faber import FaberBasisGenerator
from experiment.experiment_executor import ExperimentExecutor
from function.type import FunctionType
from grid.generator.chebyshev_grid_generator import ChebyshevGridGenerator
from grid.generator.rule_grid_generator import RuleGridGenerator
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.generator.uniform_number_generator import UniformNumberGenerator
from plot.plot_error_distribution import plot_all_errors_fixed_dim, plot_all_errors_fixed_scale
from solver.cg_least_squares import ConjugateGradient_LS
from solver.cg_normal_equation import ConjugateGradient_NE
from solver.scipy_lstsq_solver import ScipyLstsqSolver
from solver.solver import Solver
import torch


def _plot_errors(results_df, save_dir, d=None, s=None, verbose=False):
    """Generate all four error-distribution plots from the live DataFrame."""
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d, save=True, latex=True, only_maximum=False, verbose=verbose)
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d, save=True, latex=True, only_maximum=True, verbose=verbose)
    # plot_all_errors_fixed_scale(df=results_df, save_dir=save_dir, scale=s, save=True, latex=True, only_maximum=False, verbose=verbose)
    # plot_all_errors_fixed_scale(df=results_df, save_dir=save_dir, scale=s, save=True, latex=True, only_maximum=True, verbose=verbose)


def main_method(folder_name: str = None):
    # ── Fixed seeds for reproducibility (do not change) ───────────────
    uniform_seed = 42
    chebyshev_seed = 43
    test_seed = 44
    function_generation_c_seed = 45
    function_generation_w_seed = 46

    seed_list = [uniform_seed, test_seed, function_generation_c_seed, function_generation_w_seed]

    # ── Experiment parameters ─────────────────────────────────────────
    multiplier_fun_ls_train = lambda x: 2 * x  # oversampling factor for LS training grids
    multiplier_fun_test = lambda x: x
    n_fun_parallel = 50  # number of random function instances per type

    store_indices = True
    use_max_scale = False  # whether to use the maximum scale for the test grid

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

    average_c = {
        FunctionType.CONTINUOUS: 1.0,
        FunctionType.CORNER_PEAK: 1.0,
        FunctionType.DISCONTINUOUS: 1.0,
        FunctionType.GAUSSIAN: 1.0,
        FunctionType.G_FUNCTION: 1.0,
        FunctionType.OSCILLATORY: 1.0,
        FunctionType.MOROKOFF_CALFISCH_1: 1.0,
        FunctionType.PRODUCT_PEAK: 1.0,
        FunctionType.ZHOU: 1.0,
        FunctionType.NOISE: 1.0,
    }

    # ── Grid generators ───────────────────────────────────────────────
    twice_points_uniform_grid_generator = UniformGridGenerator(seed=uniform_seed,
                                                               multiplier_fun=multiplier_fun_ls_train)
    twice_points_chebyshev_grid_generator = ChebyshevGridGenerator(seed=chebyshev_seed,
                                                                   multiplier_fun=multiplier_fun_ls_train)
    rule_grid_generator = RuleGridGenerator(output_dim=n_fun_parallel * len(function_types))
    test_grid_generator = UniformGridGenerator(seed=test_seed, multiplier_fun=multiplier_fun_test)

    # ── Basis generators ──────────────────────────────────────────────
    clenshawcurtis_basis_generator = ClenshawCurtisLevelPolynomialBasisGenerator(store_indices=store_indices)
    faber_basis_generator = FaberBasisGenerator()
    # Placeholder basis/solver for Smolyak (Tasmanian handles both internally)
    aux_smolyak_basis_generator = BasisGenerator("CHEBYSHEV", "CS")

    # ── Solver ────────────────────────────────────────────────────────
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    scipy_lstsq_gelsy_solver = ScipyLstsqSolver(driver='gelsy')
    aux_smolyak_solver = Solver("TASMANIAN", "TM")
    cg_ls_solver = ConjugateGradient_LS(max_iter=1000, tolerance=1e-6, device=device)
    cg_ne_solver = ConjugateGradient_NE(max_iter=1000, tolerance=1e-6, device=device)

    # ── Function parameter generators ─────────────────────────────────
    uniform_value_generator_c = UniformNumberGenerator(seed=function_generation_c_seed)
    uniform_value_generator_w = UniformNumberGenerator(seed=function_generation_w_seed)

    # ── Algorithms ────────────────────────────────────────────────────
    ls = LeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_uniform_grid_generator,
                               solver=scipy_lstsq_gelsy_solver)
    wls = WeightedLeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_chebyshev_grid_generator,
                                        solver=scipy_lstsq_gelsy_solver)
    sa = SmolyakAlgorithm(basis_generator=aux_smolyak_basis_generator, grid_generator=rule_grid_generator,
                          solver=aux_smolyak_solver)

    faber_ls = LeastSquaresAlgorithm(faber_basis_generator,
                                     twice_points_uniform_grid_generator,
                                     solver=scipy_lstsq_gelsy_solver)

    faber_wls = WeightedLeastSquaresAlgorithm(faber_basis_generator,
                                              twice_points_chebyshev_grid_generator,
                                              solver=scipy_lstsq_gelsy_solver)

    ls_cg_ne = LeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_uniform_grid_generator,
                                     solver=cg_ne_solver)
    ls_cg_ls = LeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_uniform_grid_generator,
                                     solver=cg_ls_solver)
    wls_cg_ne = WeightedLeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_chebyshev_grid_generator,
                                              solver=cg_ne_solver)
    wls_cg_ls = WeightedLeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_chebyshev_grid_generator,
                                              solver=cg_ls_solver)

    algorithm_list = [sa, ls, wls, faber_ls, faber_wls, ls_cg_ls, ls_cg_ne, wls_cg_ls, wls_cg_ne]

    if folder_name is not None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")
    else:
        path = None

    ex = ExperimentExecutor(dim_scale_dict,
                            test_grid_generator=test_grid_generator,
                            uniform_value_generator_c=uniform_value_generator_c,
                            uniform_value_generator_w=uniform_value_generator_w,
                            use_max_scale=use_max_scale,
                            path=path)

    ex.execute_experiments(algorithm_list, function_types, n_fun_parallel, avg_c=average_c, seed_list=seed_list,
                           device=device, plot_callbacks=[_plot_errors])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main method and store the results in the given folder')
    parser.add_argument('-f', '--folder_name', default=None, type=str, required=False,
                        help='The name of the folder where the results will be stored')
    args = parser.parse_args()
    main_method(folder_name=args.folder_name)
