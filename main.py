import datetime

from experiments.experiments import run_experiments
from tqdm import tqdm
from test_functions.function_types import FunctionType
from interpolate.interpolation_methods import LeastSquaresMethod, SmolyakMethod
from utils.utils import plot_errors

if __name__ == '__main__':
    dim_range = range(3, 4)
    scale_range = range(1, 8)
    methods = ['Smolyak', 'Least_Squares_Uniform', 'Least_Squares_Chebyshev_Weight']
    function_types = [FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK,
                      FunctionType.GAUSSIAN, FunctionType.CONTINUOUS, FunctionType.DISCONTINUOUS, FunctionType.G_FUNCTION]

    ls_method_type = LeastSquaresMethod.NUMPY_LSTSQ

    if ls_method_type == LeastSquaresMethod.PYTORCH_NEURAL_NET:
        function_types = [FunctionType.OSCILLATORY]

    smolyak_method_type = SmolyakMethod.STANDARD
    additional_multiplier = 1
    n_fun_parallel = 25

    print(f"Started program at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # current folder name should be equal to the date and current time
    folder_name = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')

    run_experiments(function_types, n_fun_parallel, dims=dim_range, scales=scale_range, methods=methods,
                    add_mul=additional_multiplier, ls_method=ls_method_type, smolyak_method=smolyak_method_type,
                    folder_name=folder_name)

    # save all images in results folder
    total_iterations = len(dim_range) * len(function_types)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for dim in dim_range:
            for fun_type in function_types:
                plot_errors(dim, fun_type, scale_range, additional_multiplier, save=True, folder_name=folder_name)
                pbar.update(1)
    #
    print(f"Done at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
