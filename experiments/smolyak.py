#  Created 2024. (Elias Mindlberger)
import time

import numpy as np

from function.f import Function
from function.parametrized_f import ParametrizedFunction
from function.parametrized import ParametrizedFunction
from grid.provider.rule_grid_provider import RuleGridProvider
from test_functions.function_types import FunctionType
from test_functions.functions import get_test_function


def sample_functions(f_types: list[FunctionType], n_parallel: int, n_avg_c: int, dim: int) -> list[
    ParametrizedFunction]:
    functions = list()
    n_function_types = len(f_types)
    c = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_parallel * n_avg_c, dim))
    w = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_parallel, dim))
    w = np.vstack([w] * n_avg_c)
    for i, func_type in enumerate(f_types):
        for j in range(n_parallel):
            for k in range(n_avg_c):
                index = i * n_parallel + j * n_avg_c + k
                f, c_param, w_param = get_test_function(function_type=func_type, d=dim, c=c[index, :], w=w[index, :])
                fun = ParametrizedFunction(f=f, dim=dim, upper=0.0, lower=1.0, c=c_param, w=w_param)
                functions.append(fun)
    return functions


def run(functions: list[Function]):
    assert len(functions) > 0
    start_time = time.time()
    dim = functions[0].dim  # We assume here that all the functions in this list are of the same dim!
    gp = RuleGridProvider(input_dim=dim)
    pass
