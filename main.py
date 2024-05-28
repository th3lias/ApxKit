import datetime

from experiments.experiments import run_experiments
from tqdm import tqdm
from genz.genz_function_types import GenzFunctionType
from utils.utils import plot_errors

if __name__ == '__main__':
    dim_range = range(10, 20)
    scale_range = range(1, 8)
    n_fun_parallel = 10

    print(f"Started program at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")

    run_experiments(n_fun_parallel, dims=dim_range, scales=scale_range)

    # visualize one specific instance
    # plot_errors(10, GenzFunctionType.OSCILLATORY, range(1, 5), save=True)

    # save all images in results folder
    total_iterations = len(dim_range) * len(GenzFunctionType)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for dim in dim_range:
            for fun_type in GenzFunctionType:
                plot_errors(dim, fun_type, scale_range, save=True)
                pbar.update(1)

    print(f"Done at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")
