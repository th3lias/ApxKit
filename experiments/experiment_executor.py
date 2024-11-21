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


class ExperimentExecutor:
    # TODO: Maybe this is the main class, and we don't need the other ones
    # TODO: Needs to somehow generalize such that Smolyak and Least Squares use the same functions
    # TODO: Docstring
    # TODO: Time those experiments

    def __init__(self, dim_list: list[int], scale_list: list[int], path: str, smoylak_method: InterpolationMethod,
                 least_squares_method: LeastSquaresMethod):

        self.dim_list = dim_list
        self.scale_list = scale_list
        self.smolyak_method = smoylak_method
        self.least_squares_method = least_squares_method
        # TODO: Check this comment below
        # Ensure that any used directory is created before this object is created, otherwise this fails.
        self.results_path = path
        self.header_keys = ['function', 'dim', 'scale', 'c', 'w', 'l_2', 'l_inf']  # TODO: This needs to be adapted
        header = dict.fromkeys(self.header_keys, list())
        self.functions = None
        df = pd.DataFrame(header)
        df.to_csv(self.results_path, index=False, sep=',', decimal='.')

    def execute_experiments(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                            avg_c: float = 1.0, multiplier_fun: Callable = lambda x: 2 * x, seed: int = 42):
        """
            Execute a series of experiments with the given function types.
        """

        # TODO: Seed necessary?

        # TODO: Time these experiments

        for dim in self.dim_list:

            sparse_grid_provider = RuleGridProvider(input_dim=dim, lower_bound=0.0,
                                                    upper_bound=1.0)  # TODO: This is kind of a Tasmanian thing, we need to make this variable

            uniform_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                       multiplier_fun=multiplier_fun, seed=seed,
                                                       rule=RandomGridRule.UNIFORM)  # TODO: Check this
            chebyshev_grid_provider = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0,
                                                         multiplier_fun=multiplier_fun, seed=seed,
                                                         rule=RandomGridRule.CHEBYSHEV)  # TODO: Check this

            sparse_grid = None
            uniform_grid = None
            chebyshev_grid = None

            self.functions = self._get_functions(function_types, n_functions_parallel, dim,
                                                 avg_c)  # maybe no need to store that at >>self<<

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

                if self.smolyak_method == InterpolationMethod.STANDARD:
                    si = SmolyakInterpolator(sparse_grid, self.smolyak_method)
                    si.fit(self.functions)

                    y_test_hat_smolyak = si.interpolate(test_grid)


                elif self.smolyak_method == InterpolationMethod.TASMANIAN:
                    sparse_grid = RuleGridProvider.generate(scale)
                    fitter = SmolyakFitter(dim)
                    model = fitter.fit(self.functions, sparse_grid)
                    y_test_hat_smolyak = model(test_grid.grid)

                else:
                    raise ValueError("Unknown interpolation method")

                # Error calculation

                smolyak_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_smolyak)))
                smolyak_ell_infty = np.max(np.abs(y_test - y_test_hat_smolyak))

                # LEAST SQUARES UNIFORM

                ls_uniform = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV,
                                                      grid=uniform_grid, method=self.least_squares_method)
                ls_uniform.fit(self.functions)

                y_test_hat_ls_uniform = ls_uniform.interpolate(test_grid)

                ls_uniform_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_ls_uniform)))
                ls_uniform_ell_infty = np.max(np.abs(y_test - y_test_hat_ls_uniform))

                # LEAST SQUARES CHEBYSHEV

                ls_chebyshev = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV,
                                                        grid=chebyshev_grid, method=self.least_squares_method)
                ls_chebyshev.fit(self.functions)

                y_test_hat_cheby_uniform = ls_chebyshev.interpolate(test_grid)

                ls_cheby_ell_2 = np.sqrt(np.mean(np.square(y_test - y_test_hat_cheby_uniform)))
                ls_cheby_ell_infty = np.max(np.abs(y_test - y_test_hat_cheby_uniform))

        print("Done")

    def _get_functions(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                       dim: int, avg_c: float) -> List[Function]:
        """
            Get a list of functions of the given types and dimension.
        """

        if isinstance(function_types, FunctionType):
            function_types = [function_types]

        functions = []

        for fun_type in function_types:
            for i in range(n_functions_parallel):
                # get c and w
                c, w = self._get_c_and_w(n_functions_parallel, avg_c, dim)
                function = ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w)
                functions.append(function)

        return functions

    def _get_c_and_w(self, n_fun_parallel: float, avg_c: float, dim: int):
        """
            Get c and w for the functions.
        """

        w = np.random.uniform(low=0.0, high=1.0, size=dim)
        c = np.random.uniform(low=0.0, high=2.0, size=dim)

        # normalize c
        c = c / np.sum(c) * dim * avg_c

        return c, w

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
    # Test the impmlementation in a small setting
    dim_list = [3, 4, 5]
    scale_list = [1, 2, 3, 4]
    path = "test.csv"
    ee = ExperimentExecutor(dim_list, scale_list, path, InterpolationMethod.STANDARD,
                            LeastSquaresMethod.SCIPY_LSTSQ_GELSY)
    ee.execute_experiments([FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK], 15, 1.0,
                           lambda x: 2 * x, 42)
