import datetime
from experiments.experiments import run_experiments
from tqdm import tqdm
from function.type import FunctionType
from fit.method.least_squares_method import LeastSquaresMethod
from fit.method.interpolation_method import InterpolationMethod
from utils.utils import plot_errors
from typing import Union
import argparse

# TODO [Jakob] Check if load and save indices and grids is implemented if possible (everywhere)

def main_method(folder_name: Union[str, None] = None):
    dim_range = range(3, 4)
    scale_range = range(1, 5)
    methods = ['Smolyak', 'Least_Squares_Uniform', 'Least_Squares_Chebyshev_Weight']
    function_types = [FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK,
                      FunctionType.GAUSSIAN, FunctionType.CONTINUOUS, FunctionType.DISCONTINUOUS,
                      FunctionType.G_FUNCTION, FunctionType.MOROKOFF_CALFISCH_1, FunctionType.MOROKOFF_CALFISCH_2,
                      FunctionType.ROOS_ARNOLD, FunctionType.BRATLEY, FunctionType.ZHOU]

    realization_seeds = [42]  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    average_c = [1]
    smolyak_method_type = InterpolationMethod.STANDARD
    additional_multiplier = 2

    multiplier_fun = lambda x: additional_multiplier * x # TODO[Jakob]: New method (from Elias) might not adapt this behavior

    n_fun_parallel = 20

    ls_method_type = LeastSquaresMethod.NUMPY_LSTSQ # TODO[Jakob]: Make something similar also for Smolyak

    current_datetime = datetime.datetime.now()

    print(f"Started program at {current_datetime.strftime('%d/%m/%Y %H:%M:%S')}")

    # current folder name should be equal to the date and current time
    if folder_name is None:
        folder_name = current_datetime.strftime('%d_%m_%Y_%H_%M')

    error = False

    try:
        run_experiments(function_types, n_fun_parallel, seed_realizations=realization_seeds, dims=dim_range,
                        scales=scale_range, methods=methods, average_c=average_c,
                        ls_method=ls_method_type, smolyak_method=smolyak_method_type,
                        folder_name=folder_name, multiplier_fun=multiplier_fun)
    except MemoryError as e:
        error = True
        print(f"Memory error occured at {current_datetime.strftime('%d/%m/%Y %H:%M:%S')} with message {e}")

    # TODO [Jakob]: Rethink if this is necessary, as in the debugging process it is not useful
    # except Exception as e:
    #     error = True
    #     print(f"Unknown error occured at {current_datetime.strftime('%d/%m/%Y %H:%M:%S')} with message {e}")

    if not error:
        # save all images in results folder
        total_iterations = len(dim_range) * len(function_types) * len(realization_seeds)
        with tqdm(total=total_iterations, desc="Plotting the results") as pbar:
            for dim in dim_range:
                for fun_type in function_types:
                    for seed in realization_seeds:
                        plot_errors(dim, seed, fun_type, scale_range, multiplier_fun, save=True,
                                    folder_name=folder_name, same_axis_both_plots=True)
                        pbar.update(1)

    print(f"Done at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main method and store the results in the given folder')
    parser.add_argument('-f', '--folder_name', default=None, type=str, required=False,
                        help='The name of the folder where the results will be stored')
    args = parser.parse_args()

    main_method(folder_name=args.folder_name)
