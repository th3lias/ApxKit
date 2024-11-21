import datetime
import os.path
import time

import numpy as np
import pandas as pd
from numpy import flatiter

from fit import InterpolationMethod, LeastSquaresMethod, SmolyakFitter, BasisType
from function import FunctionType, Function, ParametrizedFunctionProvider
from typing import Union, List, Callable

from grid import RuleGridProvider
from grid.provider.random_grid_provider import RandomGridProvider
from grid.rule.random_grid_rule import RandomGridRule
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import calculate_num_points


class ExperimentExecutor:
    # TODO: Maybe this is the main class, and we don't need the other ones
    # TODO: Needs to somehow generalize such that Smolyak and Least Squares use the same functions
    # TODO: Docstring
    # TODO: Progressbar necessary
    # TODO: Adapt to the TASMANIAN API but not necessarily use the tasmanian API

    def __init__(self, dim_list: list[int], scale_list: list[int], smoylak_method: InterpolationMethod,
                 least_squares_method: LeastSquaresMethod, path: str = None):

        current_datetime = datetime.datetime.now()

        if path is None:
            self.results_path = os.path.join("results", current_datetime.strftime('%d_%m_%Y_%H_%M_%S'),
                                "results_numerical_experiments.csv")

        self.dim_list = dim_list
        self.scale_list = scale_list
        self.smolyak_method = smoylak_method
        self.least_squares_method = least_squares_method
        # TODO: Check this comment below
        # Ensure that any used directory is created before this object is created, otherwise this fails.

        self.header_keys = ['dim', 'scale', 'method', 'w', 'c', 'sum_c', 'grid_type', 'basis_type', 'method_type',
                            'n_samples', 'seed', 'test_grid_seed', 'f_name', 'ell_2_error', 'ell_infty_error',
                            'datetime', 'needed_time']  # TODO: This needs to be adapted
        header = dict.fromkeys(self.header_keys, list())
        self.functions = None
        df = pd.DataFrame(header)

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        df.to_csv(self.results_path, index=False, sep=',', decimal='.', header=True)

    def execute_experiments(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                            avg_c: float = 1.0, ls_multiplier_fun: Callable = lambda x: 2 * x, seed: int = 42):
        """
            Execute a series of experiments with the given function types.
        """

        # TODO: Seed necessary?

        # TODO: Time these experiments

        for dim in self.dim_list:

            sparse_grid_provider = RuleGridProvider(input_dim=dim, lower_bound=0.0,
                                                    upper_bound=1.0)  # TODO: This is kind of a Tasmanian thing, we need to make this variable

            uniform_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                       multiplier_fun=ls_multiplier_fun, seed=seed,
                                                       rule=RandomGridRule.UNIFORM)  # TODO: Check this
            chebyshev_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                         multiplier_fun=ls_multiplier_fun, seed=seed,
                                                         rule=RandomGridRule.CHEBYSHEV)  # TODO: Check this

            sparse_grid = None
            uniform_grid = None
            chebyshev_grid = None

            self.functions, self.cs, self.ws, self.f_names = self._get_functions(function_types, n_functions_parallel,
                                                                                 dim,
                                                                                 avg_c)  # TODO: maybe no need to store that at >>self<<

            for scale in self.scale_list:

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

                test_grid = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0).generate(scale)
                n_points = test_grid.get_num_points()
                y_test = np.empty(dtype=np.float64, shape=(len(self.functions), n_points))

                for i, function in enumerate(self.functions):
                    y_test[i] = function(test_grid.grid)

                # TODO: Maybe make the same for the train grid and get y from the function as otherwise it will be called several time although it is the same calculation

                # SMOLYAK

                start_time = time.time()

                if self.smolyak_method == InterpolationMethod.STANDARD:
                    si = SmolyakInterpolator(sparse_grid, self.smolyak_method)
                    si.fit(self.functions)

                    y_test_hat_smolyak = si.interpolate(test_grid)


                elif self.smolyak_method == InterpolationMethod.TASMANIAN:
                    fitter = SmolyakFitter(dim)
                    y_test_hat_smolyak = np.empty(dtype=np.float64, shape=(len(self.functions), n_points))
                    for i, function in enumerate(self.functions):
                        model = fitter.fit(function, sparse_grid) # TODO: Check if this is possible for multiple in parallel
                        y_test_hat_smolyak[i] = model(test_grid.grid).squeeze()

                else:
                    raise ValueError("Unknown interpolation method")

                # Error calculation

                smolyak_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_smolyak), axis=1))
                smolyak_ell_infty = np.max(np.abs(y_test - y_test_hat_smolyak), axis=1)

                end_time = time.time()
                needed_time = end_time - start_time
                cur_datetime = datetime.datetime.now()

                self.save_stats(
                    dim=dim,
                    scale=scale,
                    method="Smolyak",
                    grid_type="CHEBYSHEV",
                    basis_type="CHEBYSHEV",
                    multiplier_fun=lambda x: x,
                    seed=seed,
                    test_grid_seed=seed,  # TODO: Probably not correct
                    ell_2_errors=smolyak_ell_2,
                    ell_infty_errors=smolyak_ell_infty,
                    datetime=cur_datetime,
                    needed_time=needed_time
                )

                # LEAST SQUARES UNIFORM

                start_time = time.time()

                ls_uniform = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV,
                                                      grid=uniform_grid, method=self.least_squares_method)
                ls_uniform.fit(self.functions)

                y_test_hat_ls_uniform = ls_uniform.interpolate(test_grid)

                # TODO: This could be generalized (also for the following error calculations
                ls_uniform_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_ls_uniform), axis=1))
                ls_uniform_ell_infty = np.max(np.abs(y_test - y_test_hat_ls_uniform), axis=1)

                end_time = time.time()
                needed_time = end_time - start_time
                cur_datetime = datetime.datetime.now()

                self.save_stats(
                    dim=dim,
                    scale=scale,
                    method="Least_Squares",
                    grid_type=uniform_grid_provider.rule.name, # Check if that corresponds to some sort of Least-Squares Random Uniform
                    basis_type=ls_uniform.basis_type.name,
                    multiplier_fun=lambda x: x,
                    seed=seed,
                    test_grid_seed=seed,  # TODO: Probably not correct
                    ell_2_errors=ls_uniform_ell_2,
                    ell_infty_errors=ls_uniform_ell_infty,
                    datetime=cur_datetime,
                    needed_time=needed_time
                )

                # LEAST SQUARES CHEBYSHEV

                start_time = time.time()

                ls_chebyshev = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV,
                                                        grid=chebyshev_grid, method=self.least_squares_method)
                ls_chebyshev.fit(self.functions)

                y_test_hat_cheby_uniform = ls_chebyshev.interpolate(test_grid)

                ls_cheby_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_cheby_uniform), axis=1))
                ls_cheby_ell_infty = np.max(np.abs(y_test - y_test_hat_cheby_uniform), axis=1)

                end_time = time.time()
                needed_time = end_time - start_time
                cur_datetime = datetime.datetime.now()

                self.save_stats(
                    dim=dim,
                    scale=scale,
                    method="Least_Squares",
                    grid_type=chebyshev_grid_provider.rule.name,
                    # Check if that corresponds to some sort of Least-Squares Random Uniform
                    basis_type=ls_chebyshev.basis_type.name,
                    multiplier_fun=lambda x: x,
                    seed=seed,
                    test_grid_seed=seed,  # TODO: Probably not correct
                    ell_2_errors=ls_cheby_ell_2,
                    ell_infty_errors=ls_cheby_ell_infty,
                    datetime=cur_datetime,
                    needed_time=needed_time
                )

        print("Done")
        # TODO: Add more statistics: Progressbar, RAM, time, datetime, etc.

    def _get_functions(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                       dim: int, avg_c: float) -> (List[Function], List[np.ndarray], List[np.ndarray], List[str]):
        """
            Get a list of functions of the given types and dimension.
        """

        if isinstance(function_types, FunctionType):
            function_types = [function_types]

        functions = []

        cs = list()
        ws = list()
        f_names = list()

        for fun_type in function_types:
            for i in range(n_functions_parallel):
                # get c and w
                c, w = self._get_c_and_w(n_functions_parallel, avg_c, dim)
                function = ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w)
                cs.append(c)
                ws.append(w)
                f_names.append(
                    fun_type.name)  # TODO: Maybe adapt the name of the functions by just using the name of the enum
                functions.append(function)

        return functions, cs, ws, f_names

    def _get_c_and_w(self, n_fun_parallel: float, avg_c: float, dim: int):
        """
            Get c and w for the functions.
        """

        w = np.random.uniform(low=0.0, high=1.0, size=dim)
        c = np.random.uniform(low=0.0, high=2.0, size=dim)

        # normalize c
        c = c / np.sum(c) * dim * avg_c

        return c, w

    def save_stats(self, dim: int, scale: int, method: str, grid_type: str, basis_type: str, multiplier_fun: Callable, seed: int, test_grid_seed: int,
                   ell_2_errors: Union[np.ndarray, List[float]],
                   ell_infty_errors: Union[np.ndarray, List[float]], datetime: datetime.datetime, needed_time: float):
        """
            Keep the CSV up to date with the current results.
        """

        # TODO: For one specific c-value we got the scientific notation, which was then stored as a string in the csv


        data = dict.fromkeys(self.header_keys, list())

        if isinstance(ell_2_errors, np.ndarray):
            ell_2_errors = ell_2_errors.tolist()
        if isinstance(ell_infty_errors, np.ndarray):
            ell_infty_errors = ell_infty_errors.tolist()

        n = len(ell_2_errors)

        if method == "Smolyak": # TODO: Maybe make this more waterproof by not comparing raw strings
            multiplier_fun = lambda x: x
            method_type = self.smolyak_method.name
            basis_type = "CHEBYSHEV"
        else:
            method_type = self.least_squares_method.name
            grid_type = ""

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
        data['test_grid_seed'] = [test_grid_seed] * n
        data['f_name'] = self.f_names
        data['ell_2_error'] = ell_2_errors
        data['ell_infty_error'] = ell_infty_errors
        data['datetime'] = [datetime] * n
        data['needed_time'] = [needed_time] * n

        df = pd.DataFrame(data)
        df.to_csv(self.results_path, mode='a', header=False, index=False)

    # # TODO: Maybe adapt the methods in here
    # def execute_random_experiments(self, function_types: list[FunctionType]):
    #     """
    #         Execute a series of experiments with random functions.
    #     """
    #     raise NotImplementedError()
    #
    # def execute_single_dim_experiment(self, function_types: list[FunctionType]):
    #     """
    #         Execute a series of experiments with functions of a single dimension.
    #     """
    #     raise NotImplementedError()


if __name__ == '__main__':
    # TODO: Try out Tasmanian method
    # Test the impmlementation in a small setting
    dim_list = [2, 3, 4, 5]
    scale_list = [1, 2, 3, 4]
    path = None
    ee = ExperimentExecutor(dim_list, scale_list, InterpolationMethod.TASMANIAN,
                            LeastSquaresMethod.SCIPY_LSTSQ_GELSY, path)
    ee.execute_experiments([FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK], 15, 1.0,
                           lambda x: 2 * x, 42)
