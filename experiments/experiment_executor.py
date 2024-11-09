import pandas as pd

from function import FunctionType, Function, ParametrizedFunctionProvider
from typing import Union, List

class ExperimentExecutor:
    # TODO: Maybe this is the main class, and we donÄt
    # TODO: Needs to somehow generalize such that Smolyak and Least Squares use the same functions
    # TODO: Docstring

    def __init__(self, dim_list: list[int], scale_list: list[int], num_test_points: int, path: str):

        """
            Abstract Class that is later inherited by the SmolyakExperimentExecutor and LeastSquaresExperimentExecutor.
        """

        self.dim_list = dim_list
        self.scale_list = scale_list
        self.num_test_points = num_test_points
        # TODO: Check this comment below
        # Ensure that any used directory is created before this object is created, otherwise this fails.
        self.results_path = path
        self.header_keys = ['function', 'dim', 'scale', 'c', 'w', 'l_2', 'l_inf']
        header = dict.fromkeys(self.header_keys, list())
        df = pd.DataFrame(header)
        df.to_csv(self.results_path, index=False, sep=',', decimal='.')

    def execute_experiments(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel:int):
        """
            Execute a series of experiments with the given function types.
        """

        for dim in self.dim_list:
            functions = self._get_functions(function_types, n_functions_parallel, dim)
            # maybe calculate y or pass the parameterized functions

            # continue here

        raise NotImplementedError()



    def _get_functions(self, function_types: Union[List[FunctionType], FunctionType], n_functions_parallel:int, dim: int)-> List[Function]:
        """
            Get a list of functions of the given types and dimension.
        """

        if isinstance(function_types, FunctionType):
            function_types = [function_types]

        functions = []


        for fun_type in function_types:
            for i in range(n_functions_parallel):
                # get c and w
                c, w = self._get_c_and_w()
                function = ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w)
                functions.append(function)


        return functions

    def _get_c_and_w(self):
        raise NotImplementedError()
        return 1,1



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

