import argparse
import datetime
import os

from algorithm.least_squares import LeastSquaresAlgorithm
from algorithm.omp import OMP, IndexSetType
from algorithm.smolyak import SmolyakAlgorithm
from algorithm.weighted_least_squares import WeightedLeastSquaresAlgorithm
from basis.basis_generator import BasisGenerator
from basis.clenshaw_curtis_level_polynomial_basis_generator import ClenshawCurtisLevelPolynomialBasisGenerator
from experiment.experiment_executor import ExperimentExecutor
from function.type import FunctionType
from grid.generator.chebyshev_grid_generator import ChebyshevGridGenerator
from grid.generator.rule_grid_generator import RuleGridGenerator
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.generator.uniform_number_generator import UniformNumberGenerator
from plot.plot_error_distribution import plot_all_errors_fixed_dim, plot_all_errors_fixed_scale
from solver.torch_omp_solver import TorchOMPSolver
from solver.scipy_lstsq_solver import ScipyLstsqSolver
from solver.solver import Solver
import torch


def _plot_errors(results_df, save_dir, d=None, s=None, verbose=False):
    """Generate all four error-distribution plots from the live DataFrame."""
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d, save=True, latex=True, only_maximum=False,
                              verbose=verbose)
    plot_all_errors_fixed_dim(df=results_df, save_dir=save_dir, dim=d, save=True, latex=True, only_maximum=True,
                              verbose=verbose)
    # plot_all_errors_fixed_scale(df=results_df, save_dir=save_dir, scale=s, save=True, latex=True, only_maximum=False, verbose=verbose)
    # plot_all_errors_fixed_scale(df=results_df, save_dir=save_dir, scale=s, save=True, latex=True, only_maximum=True, verbose=verbose)


def main_method(folder_name: str = None):
    # ── Fixed seeds for reproducibility (do not change) ───────────────

    seeds = {
        "uniform_seed": 42,
        "chebyshev_seed": 43,
        "test_seed": 44,
        "function_generation_c_seed": 45,
        "function_generation_w_seed": 46
    }
    seed_list = [seeds["uniform_seed"], seeds["chebyshev_seed"], seeds["test_seed"],
                 seeds["function_generation_c_seed"], seeds["function_generation_w_seed"]]

    # ── Experiment parameters ─────────────────────────────────────────
    multiplier_fun_ls_train = lambda x: 2 * x  # oversampling factor for LS training grids
    multiplier_fun_test = lambda x: 1 * x
    n_fun_parallel = 5  # number of random function instances per type

    store_indices = True
    use_max_scale = True  # whether to use the maximum scale for the test grid

    dim_scale_dict = {
        2: [1, 2],
        # 3: [1, 2],
        # 4: [1, 2, 3, 4, 5],
        # 5: [1, 2, 3, 4, 5, ],
        # 6: [1, 2, 3, 4, ],
        # 7: [1, 2, 3, 4, ],
        # 8: [1, 2, 3, 4, ],
        # 9: [1, 2, 3, 4, ],
        # 10: [1, 2, 3, 4, ],
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
    twice_points_uniform_grid_generator = UniformGridGenerator(seed=seeds['uniform_seed'],
                                                               multiplier_fun=multiplier_fun_ls_train)
    twice_points_chebyshev_grid_generator = ChebyshevGridGenerator(seed=seeds['chebyshev_seed'],
                                                                   multiplier_fun=multiplier_fun_ls_train)

    rule_grid_generator = RuleGridGenerator(output_dim=n_fun_parallel * len(function_types))  # for Tasmanian
    test_grid_generator = UniformGridGenerator(seed=seeds['test_seed'], multiplier_fun=multiplier_fun_test)

    # ── Basis generators ──────────────────────────────────────────────
    clenshawcurtis_basis_generator = ClenshawCurtisLevelPolynomialBasisGenerator(store_indices=store_indices)

    # Placeholder basis/solver for Smolyak (Tasmanian handles both internally)
    aux_smolyak_basis_generator = BasisGenerator("CHEBYSHEV", "CS")

    # ── Solver ────────────────────────────────────────────────────────
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    scipy_lstsq_gelsy_solver = ScipyLstsqSolver(driver='gelsy')
    omp_solver = TorchOMPSolver(num_iters=20_000, tol=1e-6, device=device)
    aux_smolyak_solver = Solver("TASMANIAN", "TM")

    # ── Function parameter generators ─────────────────────────────────
    uniform_value_generator_c = UniformNumberGenerator(seed=seeds['function_generation_c_seed'])
    uniform_value_generator_w = UniformNumberGenerator(seed=seeds['function_generation_w_seed'])

    # ── Algorithms ────────────────────────────────────────────────────
    ls = LeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_uniform_grid_generator,
                               solver=scipy_lstsq_gelsy_solver)
    wls = WeightedLeastSquaresAlgorithm(clenshawcurtis_basis_generator, twice_points_chebyshev_grid_generator,
                                        solver=scipy_lstsq_gelsy_solver)

    omp_uniform_grid_hyperbolic_2 = OMP(grid_generator=twice_points_uniform_grid_generator,
                                        basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                        solver=omp_solver,
                                        device=device,
                                        hc_bandwidth=None,
                                        index_set_type=IndexSetType.HYPERBOLIC,
                                        name="Orthonal_Matching_Pursuit_Uniform_Hyperbolic_2",
                                        abbr_name="OMP_Unif_hyp_2",
                                        bandwidth_multiplier_function=lambda x: 2 * x)

    omp_uniform_grid_hyperbolic_5 = OMP(grid_generator=twice_points_uniform_grid_generator,
                                        basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                        solver=omp_solver,
                                        device=device,
                                        hc_bandwidth=None,
                                        index_set_type=IndexSetType.HYPERBOLIC,
                                        name="Orthonal_Matching_Pursuit_Uniform_Hyperbolic_5",
                                        abbr_name="OMP_Unif_hyp_5",
                                        bandwidth_multiplier_function=lambda x: 5 * x)

    omp_uniform_grid_hyperbolic_10 = OMP(grid_generator=twice_points_uniform_grid_generator,
                                         basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                         solver=omp_solver,
                                         device=device,
                                         hc_bandwidth=None,
                                         index_set_type=IndexSetType.HYPERBOLIC,
                                         name="Orthonal_Matching_Pursuit_Uniform_Hyperbolic_10",
                                         abbr_name="OMP_Unif_hyp_10",
                                         bandwidth_multiplier_function=lambda x: 10 * x)

    omp_uniform_grid_hyperbolic_20 = OMP(grid_generator=twice_points_uniform_grid_generator,
                                         basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                         solver=omp_solver,
                                         device=device,
                                         hc_bandwidth=None,
                                         index_set_type=IndexSetType.HYPERBOLIC,
                                         name="Orthonal_Matching_Pursuit_Uniform_Hyperbolic_20",
                                         abbr_name="OMP_Unif_hyp_20",
                                         bandwidth_multiplier_function=lambda x: 20 * x)

    omp_chebyshev_grid_hyperbolic_2 = OMP(grid_generator=twice_points_chebyshev_grid_generator,
                                          basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                          solver=omp_solver,
                                          device=device,
                                          hc_bandwidth=None,
                                          index_set_type=IndexSetType.HYPERBOLIC,
                                          name="Orthonal_Matching_Pursuit_Chebyshev_Hyperbolic_2",
                                          abbr_name="OMP_Cheb_hyp_2",
                                          bandwidth_multiplier_function=lambda x: 2 * x)

    omp_chebyshev_grid_hyperbolic_5 = OMP(grid_generator=twice_points_chebyshev_grid_generator,
                                          basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                          solver=omp_solver,
                                          device=device,
                                          hc_bandwidth=None,
                                          index_set_type=IndexSetType.HYPERBOLIC,
                                          name="Orthonal_Matching_Pursuit_Chebyshev_Hyperbolic_5",
                                          abbr_name="OMP_Cheb_hyp_5",
                                          bandwidth_multiplier_function=lambda x: 5 * x)

    omp_chebyshev_grid_hyperbolic_10 = OMP(grid_generator=twice_points_chebyshev_grid_generator,
                                           basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                           solver=omp_solver,
                                           device=device,
                                           hc_bandwidth=None,
                                           index_set_type=IndexSetType.HYPERBOLIC,
                                           name="Orthonal_Matching_Pursuit_Chebyshev_Hyperbolic_10",
                                           abbr_name="OMP_Cheb_hyp_10",
                                           bandwidth_multiplier_function=lambda x: 10 * x)

    omp_chebyshev_grid_hyperbolic_20 = OMP(grid_generator=twice_points_chebyshev_grid_generator,
                                           basis_generator=clenshawcurtis_basis_generator,  # TODO: Not used now
                                           solver=omp_solver,
                                           device=device,
                                           hc_bandwidth=None,
                                           index_set_type=IndexSetType.HYPERBOLIC,
                                           name="Orthonal_Matching_Pursuit_Chebyshev_Hyperbolic_20",
                                           abbr_name="OMP_Cheb_hyp_20",
                                           bandwidth_multiplier_function=lambda x: 20 * x)

    sa = SmolyakAlgorithm(basis_generator=aux_smolyak_basis_generator, grid_generator=rule_grid_generator,
                          solver=aux_smolyak_solver)

    algorithm_list = [sa, ls, wls,
                      omp_uniform_grid_hyperbolic_2,
                      omp_uniform_grid_hyperbolic_5,
                      omp_uniform_grid_hyperbolic_10,
                      omp_uniform_grid_hyperbolic_20,
                      omp_chebyshev_grid_hyperbolic_2,
                      omp_chebyshev_grid_hyperbolic_5,
                      omp_chebyshev_grid_hyperbolic_10,
                      omp_chebyshev_grid_hyperbolic_20]

    if folder_name is not None:
        run_dir = os.path.join("results", folder_name)
    else:
        run_dir = os.path.join("results", datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S'))
    path = os.path.join(run_dir, "results_numerical_experiments.csv")

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
