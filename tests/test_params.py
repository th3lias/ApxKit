import itertools
import multiprocessing
import time

import numpy as np

from genz.genz_functions import GenzFunctionType
from genz.genz_functions import get_genz_function
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.least_squares import LeastSquaresInterpolator


def test_configuration_iterative(param_dict, q):
    start_time = time.time()
    w = param_dict.get('w')
    c = param_dict.get('c')
    del param_dict['w']
    del param_dict['c']
    n_samples = param_dict.get('n_samples')
    del param_dict['n_samples']
    f = get_genz_function(GenzFunctionType.OSCILLATORY, w=w, c=c, d=param_dict['dim'])
    lsq = LeastSquaresInterpolator(degree=param_dict['degree'], include_bias=param_dict['include_bias'],
                                   grid=param_dict['grid'],
                                   self_implemented=True, iterative=True)
    lsq.interpolate(f=f)
    end_time = time.time()
    execution_time = end_time - start_time
    del param_dict['grid']
    del param_dict['include_bias']
    param_dict['n_samples'] = n_samples
    print(f"Done iterative with parameters {param_dict}")

    q.put((param_dict, execution_time))


def test_configuration(param_dict, q):
    start_time = time.time()
    w = param_dict.get('w')
    c = param_dict.get('c')
    del param_dict['w']
    del param_dict['c']
    n_samples = param_dict.get('n_samples')
    del param_dict['n_samples']
    f = get_genz_function(GenzFunctionType.OSCILLATORY, w=w, c=c, d=param_dict['dim'])
    lsq = LeastSquaresInterpolator(degree=param_dict['degree'], include_bias=param_dict['include_bias'],
                                   grid=param_dict['grid'],
                                   self_implemented=True, iterative=False)
    lsq.interpolate(f=f)
    end_time = time.time()
    execution_time = end_time - start_time
    del param_dict['grid']
    del param_dict['include_bias']
    param_dict['n_samples'] = n_samples
    print(f"Done with parameters {param_dict}")

    q.put((param_dict, execution_time))


def grid_search(params, timeout, filename, iterative):
    results = []
    with open(filename, "w") as f:
        f.write("degree,dim,n_samples,time\n")

    q = multiprocessing.Queue()
    for param_combo in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), param_combo))

        c = np.random.uniform(low=0, high=1, size=(param_dict['dim']))
        w = np.random.uniform(low=0, high=1, size=(param_dict['dim']))

        param_dict['w'] = w
        param_dict['c'] = c

        grid = GridProvider(param_dict['dim']).generate(GridType.RANDOM, scale=param_dict['n_samples'])

        param_dict['grid'] = grid
        param_dict['include_bias'] = False

        if iterative:
            p = multiprocessing.Process(target=test_configuration_iterative, args=(param_dict, q))
        else:
            p = multiprocessing.Process(target=test_configuration, args=(param_dict, q))

        p.start()
        p.join(timeout=timeout)
        if p.is_alive():
            del param_dict['grid']
            del param_dict['include_bias']
            del param_dict['w']
            del param_dict['c']
            param_dict['n_samples'] = param_dict['n_samples']
            with open(filename, "a") as f:
                f.write(
                    f"{param_dict['degree']},{param_dict['dim']},{param_dict['n_samples']},took longer than {timeout} "
                    f"seconds\n")
            print(f"Parameters: {param_dict}, took longer than {timeout} seconds")
            p.terminate()
            p.join()

    while not q.empty():
        result = q.get()
        param, exec_time = result
        with open(filename, "a") as f:
            f.write(f"{param['degree']},{param['dim']},{param['n_samples']},{exec_time}\n")
        print(f"Parameters: {param}, Time: {exec_time}")
        results.append((param, exec_time))

    return results


if __name__ == '__main__':
    parameters = dict(degree=[1, 2, 3], dim=[5, 10, 15, 20, 25, 30], n_samples=[1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])

    # Run grid search
    grid_results = grid_search(parameters, 150, filename="grid_search_results.txt", iterative=False)
    grid_results_iterative = grid_search(parameters, 150, filename="grid_search_results_iterative.txt", iterative=True)
