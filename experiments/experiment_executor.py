import datetime
import os.path
import platform
import time
from typing import Union, List, Callable

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from fit import InterpolationMethod, LeastSquaresMethod, SmolyakFitter, BasisType
from function import FunctionType, Function, ParametrizedFunctionProvider
from grid import RuleGridProvider, TasmanianGridType
from grid.provider.random_grid_provider import RandomGridProvider
from grid.rule.random_grid_rule import RandomGridRule
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import calculate_num_points


class ExperimentExecutor:
    """
        Runs the experiments, where Smolyak and Least Squares are compared
    """

    def __init__(self, dim_scale_dict: dict[int, List[int]], smolyak_method: InterpolationMethod,
                 least_squares_method: LeastSquaresMethod, ls_basis_type: BasisType, seed: int = None,
                 path: str = None, tasmanian_grid_type: TasmanianGridType = TasmanianGridType.STANDARD_GLOBAL,
                 store_indices: bool = True):
        current_datetime = datetime.datetime.now()
        if path is None:
            self.results_path = os.path.join("results", current_datetime.strftime('%d_%m_%Y_%H_%M_%S'),
                                             "results_numerical_experiments.csv")
        else:
            self.results_path = path

        for dim in dim_scale_dict.keys():
            dim_scale_dict[dim] = sorted(list(set(dim_scale_dict[dim])))
        self.dim_scale_dictionary = dim_scale_dict
        self.smolyak_method = smolyak_method
        self.least_squares_method = least_squares_method
        self.seed = seed
        self.least_squares_basis_type = ls_basis_type
        self.tasmanian_grid_type = tasmanian_grid_type

        self.header_keys = ['dim', 'scale', 'method', 'w', 'c', 'sum_c', 'grid_type', 'basis_type', 'method_type',
                            'n_samples', 'seed', 'f_name', 'ell_2_error', 'ell_infty_error', 'datetime', 'needed_time']
        header = dict.fromkeys(self.header_keys, list())
        self.functions = None
        self.test_functions = None
        self.cs = None
        self.ws = None
        self.f_names = None
        self.test_grid = None
        self.y_test = None
        self.store_indices = store_indices
        df = pd.DataFrame(header)
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        df.to_csv(self.results_path, index=False, sep=',', decimal='.', header=True)

    def execute_experiments(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                            avg_c: Union[float, dict], ls_multiplier_fun: Callable = lambda x: 2 * x, ):
        """
            Execute a series of comparisons with the given function types.
        """

        np.random.seed(self.seed)

        print(
            f"Starting dimension/scale {self.dim_scale_dictionary} n_functions={n_functions_parallel} "
            f"experiments with cpu {platform.processor()} and "
            f"{psutil.virtual_memory().total / 1024 / 1024 / 1024} GB RAM at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"Results will be stored in {self.results_path}")
        print("_" * 75)
        print("")

        time.sleep(1)

        n_iterations = 0

        for scales in self.dim_scale_dictionary.values():
            n_iterations += len(scales)

        total_iterations = n_iterations * 3  # 3 methods (LS_Unif, LS_Cheby, Smolyak)
        progress_bar = tqdm(total=total_iterations, desc="Initializing", unit="iteration")
        for dim in self.dim_scale_dictionary.keys():

            sparse_grid_provider = RuleGridProvider(input_dim=dim, lower_bound=0.0, upper_bound=1.0,
                                                    output_dim=len(function_types) * n_functions_parallel,
                                                    tasmanian_type=self.tasmanian_grid_type)
            uniform_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                       multiplier_fun=ls_multiplier_fun, seed=self.seed,
                                                       rule=RandomGridRule.UNIFORM)
            chebyshev_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                         multiplier_fun=ls_multiplier_fun, seed=self.seed,
                                                         rule=RandomGridRule.CHEBYSHEV)

            sparse_grid = None
            uniform_grid = None
            chebyshev_grid = None

            # Calculates the functions, their names, cs and ws and stores them on class--level
            self._get_functions(function_types, n_functions_parallel, dim, avg_c)

            for scale in self.dim_scale_dictionary.get(dim):

                # Training Grids
                if sparse_grid is None:
                    sparse_grid = sparse_grid_provider.generate(scale)
                else:
                    sparse_grid = sparse_grid_provider.increase_scale(sparse_grid, 1)

                if uniform_grid is None:
                    uniform_grid = uniform_grid_provider.generate(scale)
                else:
                    uniform_grid = uniform_grid_provider.increase_scale(uniform_grid, 1)

                if chebyshev_grid is None:
                    chebyshev_grid = chebyshev_grid_provider.generate(scale)
                else:
                    chebyshev_grid = chebyshev_grid_provider.increase_scale(chebyshev_grid, 1)

                # Test Grid
                test_grid_seed = self.seed + 42 if self.seed is not None else None
                if test_grid_seed is not None and self.seed is not None:
                    assert not self.seed == test_grid_seed, "The seed for the test grid should be different from the training grid, otherwise uniform least squares is trained and tested on the same data"
                self.test_grid = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                    seed=test_grid_seed).generate(scale)
                n_points = self.test_grid.get_num_points()
                self.y_test = np.empty(dtype=np.float64, shape=(len(self.test_functions), n_points))

                for i, test_function in enumerate(self.test_functions):
                    self.y_test[i] = test_function(self.test_grid.grid)

                # Smolyak
                progress_bar.set_description(f"Experiment: Dim:{dim},Scale:{scale},"
                                             f"Method:Smolyak,datetime:{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

                self._run_experiment_smolyak(dim, scale, sparse_grid, lambda x: x)

                progress_bar.update(1)

                # LSQ Uniform
                progress_bar.set_description(f"Experiment: Dim:{dim},Scale:{scale},"
                                             f"Method:LS_Unif,datetime:{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                self._run_experiment_ls(dim, scale, uniform_grid, "UNIFORM", ls_multiplier_fun)
                progress_bar.update(1)

                # LSQ Chebyshev
                progress_bar.set_description(f"Experiment: Dim:{dim},Scale:{scale},"
                                             f"Method:LS_Cheb,datetime:{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                self._run_experiment_ls(dim, scale, chebyshev_grid, "CHEBYSHEV", ls_multiplier_fun)
                progress_bar.update(1)

        progress_bar.close()
        print(f"Done at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    def _run_experiment_smolyak(self, dim, scale, grid, multiplier_fun: Callable):
        start_time = time.time()

        if self.smolyak_method == InterpolationMethod.STANDARD:  # Least Squares at the Sparse Grid points
            raise DeprecationWarning(
                "The standard Smolyak method is deprecated and will be removed in future versions as it does not satisfy the desired numerical accuracy.")
            si = SmolyakInterpolator(grid, self.smolyak_method, store_indices=self.store_indices)
            si.fit(self.functions)
            y_test_hat_smolyak = si.interpolate(self.test_grid)
        elif self.smolyak_method == InterpolationMethod.TASMANIAN:  # Smolyak with Tasmanian
            fitter = SmolyakFitter(dim)
            model = fitter.fit(self.functions, grid)
            y_test_hat_smolyak = model(self.test_grid.grid)
        else:
            raise ValueError(f"Method {self.smolyak_method} is not supported!")

        # Error calculation
        smolyak_ell_2, smolyak_ell_infty = self._calc_error(y_test_hat_smolyak)

        end_time = time.time()
        needed_time = end_time - start_time
        cur_datetime = datetime.datetime.now()

        self._save_stats(dim=dim, scale=scale, method="Smolyak", grid_type="SPARSE", basis_type="CHEBYSHEV",
                         multiplier_fun=multiplier_fun, seed=self.seed, ell_2_errors=smolyak_ell_2,
                         ell_infty_errors=smolyak_ell_infty, date_time=cur_datetime, needed_time=round(needed_time, 3))

    def _run_experiment_ls(self, dim, scale, grid, grid_type: str, multiplier_fun: Callable):
        start_time = time.time()

        ls = LeastSquaresInterpolator(include_bias=True, basis_type=self.least_squares_basis_type, grid=grid,
                                      method=self.least_squares_method, store_indices=self.store_indices)
        ls.fit(self.functions)
        y_test_hat = ls.interpolate(self.test_grid)
        ls_ell_2, ls_ell_infty = self._calc_error(y_test_hat)
        end_time = time.time()
        needed_time = end_time - start_time
        cur_datetime = datetime.datetime.now()
        self._save_stats(dim=dim, scale=scale, method="Least_Squares", grid_type=grid_type,
                         basis_type=self.least_squares_basis_type.name, multiplier_fun=multiplier_fun, seed=self.seed,
                         ell_2_errors=ls_ell_2, ell_infty_errors=ls_ell_infty, date_time=cur_datetime,
                         needed_time=round(needed_time, 3))

    def _get_functions(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                       dim: int, avg_c: Union[float, dict]) -> (List[Function], List[np.ndarray], List[np.ndarray],
                                                                List[str]):
        """
            Get a list of functions of the given types and dimension.
        """

        if isinstance(function_types, FunctionType):
            function_types = [function_types]

        functions = []
        test_functions = []

        cs = list()
        ws = list()
        f_names = list()

        for fun_type in function_types:
            if isinstance(avg_c, dict):
                if fun_type not in avg_c:
                    raise ValueError(f"Function type {fun_type} not in average c dictionary")
                avg_c_fun = avg_c[fun_type]
            else:
                avg_c_fun = avg_c

            for i in range(n_functions_parallel):
                # get c and w
                c, w = self._get_c_and_w(avg_c_fun, dim)
                function = ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w)
                test_function = ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w, test=True)
                cs.append(c)
                ws.append(w)
                f_names.append(fun_type.name)
                functions.append(function)
                test_functions.append(test_function)

        self.functions = functions
        self.test_functions = test_functions
        self.cs = cs
        self.ws = ws
        self.f_names = f_names

    @staticmethod
    def _get_c_and_w(avg_c: float, dim: int):
        """
            Get c and w for the functions.
        """

        w = np.random.uniform(low=0.0, high=1.0, size=dim)
        c = np.random.uniform(low=0.0, high=1.0, size=dim)

        # normalize c
        c = c / np.sum(c) * dim * avg_c

        return c, w

    def _save_stats(self, dim: int, scale: int, method: str, grid_type: str, basis_type: str, multiplier_fun: Callable,
                    seed: int, ell_2_errors: Union[np.ndarray, List[float]],
                    ell_infty_errors: Union[np.ndarray, List[float]], date_time: datetime.datetime, needed_time: float):
        """
            Keep the CSV up to date with the current results.
        """

        data = dict.fromkeys(self.header_keys, list())

        if isinstance(ell_2_errors, np.ndarray):
            ell_2_errors = ell_2_errors.tolist()
        if isinstance(ell_infty_errors, np.ndarray):
            ell_infty_errors = ell_infty_errors.tolist()

        n = len(ell_2_errors)

        if method == "Smolyak":
            method_type = self.smolyak_method.name
        else:
            method_type = self.least_squares_method.name

        n_points = int(multiplier_fun(calculate_num_points(scale, dim)))

        formatted_cs = [np.array2string(c, precision=5, separator=',', suppress_small=True).replace('\n', '') for c in
                        self.cs]

        formatted_ws = [np.array2string(w, precision=5, separator=',', suppress_small=True).replace('\n', '') for w in
                        self.ws]

        data['dim'] = [dim] * n
        data['scale'] = [scale] * n
        data['method'] = [method] * n
        data['w'] = formatted_ws
        data['c'] = formatted_cs
        data['sum_c'] = [round(np.sum(c), 3) for c in self.cs]
        data['grid_type'] = [grid_type] * n
        data['basis_type'] = [basis_type] * n
        data['method_type'] = [method_type] * n
        data['n_samples'] = [n_points] * n
        data['seed'] = [seed] * n
        data['f_name'] = self.f_names
        data['ell_2_error'] = ell_2_errors
        data['ell_infty_error'] = ell_infty_errors
        data['datetime'] = [date_time] * n
        data['needed_time'] = [needed_time] * n

        df = pd.DataFrame(data)
        df.to_csv(self.results_path, mode='a', header=False, index=False)

    def _calc_error(self, test_array):
        ell_2 = np.sqrt(np.mean(np.square(self.y_test - test_array), axis=1))
        ell_infty = np.max(np.abs(self.y_test - test_array), axis=1)

        return ell_2, ell_infty
