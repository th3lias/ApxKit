import numpy as np
import pandas as pd
from numpy import flatiter

from fit import InterpolationMethod, LeastSquaresMethod
from function import FunctionType, Function, ParametrizedFunctionProvider
from typing import Union, List

from grid import RuleGridProvider
from grid.provider.random_grid_provider import RandomGridProvider


class ExperimentExecutor:
    # TODO: Maybe this is the main class, and we donÄt
    # TODO: Needs to somehow generalize such that Smolyak and Least Squares use the same functions
    # TODO: Docstring
    # TODO: Time those experiments

    def __init__(self, dim_list: list[int], scale_list: list[int], path: str, smoylak_method:InterpolationMethod, least_squares_method: LeastSquaresMethod):

        """
            Abstract Class that is later inherited by the SmolyakExperimentExecutor and LeastSquaresExperimentExecutor.
        """

        self.dim_list = dim_list
        self.scale_list = scale_list
        self.smolyak_method = smoylak_method
        self.least_squares_method = least_squares_method
        # TODO: Check this comment below
        # Ensure that any used directory is created before this object is created, otherwise this fails.
        self.results_path = path
        self.header_keys = ['function', 'dim', 'scale', 'c', 'w', 'l_2', 'l_inf']
        header = dict.fromkeys(self.header_keys, list())
        self.functions = None
        df = pd.DataFrame(header)
        df.to_csv(self.results_path, index=False, sep=',', decimal='.')

    def execute_experiments(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel: int,
                            avg_c: float = 1.0):
        """
            Execute a series of experiments with the given function types.
        """

        for dim in self.dim_list:

            sparse_grid_provider = RuleGridProvider(input_dim=dim, lower_bound=0.0, upper_bound=1.0)
            sparse_grid = None

            self.functions = self._get_functions(function_types, n_functions_parallel, dim, avg_c)

            for scale in self.scale_list:
                test_grid = RandomGridProvider(dim, lower_bound=0.0, upper_bound=1.0).generate(scale)
                self.y_test = np.empty(dtype=np.float64, shape=len(self.functions))

                for i, function in enumerate(self.functions):
                    self.y_test[i] = function(test_grid.grid)

                # SMOLYAK

                if self.smolyak_method == InterpolationMethod.STANDARD:
                    pass
                elif self.smolyak_method == InterpolationMethod.TASMANIAN:
                    pass
                else:
                    raise ValueError("Unknown interpolation method")

                # LEAST SQUARES UNIFORM

                if self.least_squares_method == LeastSquaresMethod.NUMPY_LSTSQ


                # LEAST SQUARES CHEBYSHEV



                if sparse_grid is None:
                    sparse_grid = sparse_grid_provider.generate(scale)
                else:
                    sparse_grid = sparse_grid_provider.increase_scale(sparse_grid, 1)

                # Do Smolyak interpolation




            # maybe calculate y or pass the parameterized functions

            # continue here

        raise NotImplementedError()

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

        assert c.mean() != avg_c  # This needs to fail, then we can delete it

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
