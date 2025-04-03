from typing import Union, List, Callable

from function import FunctionType

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.utils import get_next_filename


def plot_errors(dimension, seed: int, function_type: FunctionType, scales: List[int], multiplier_fun: Callable,
                folder_name: str, path: Union[str, None] = None, save: bool = False,
                save_path: Union[str, None] = None, same_axis_both_plots: bool = True, latex: bool = False):
    """
    Creates plots of each different c-value for a given function type, given the path of the results-csv file.
    The ell2 and the max error are plotted.

    :param dimension: dimension which should be considered from the results file
    :param seed: Representing the realization of the algorithm.
    :param function_type: Specifies which function should be considered
    :param scales: range of scales, which are considered
    :param multiplier_fun: specifies which multiplier was used to increase the number of samples in least squares
    :param folder_name: name of the folder where the results are stored
    :param path: Path of the results-csv file. If None, a default path will be used.
    :param save: Specifies whether the images should be saved. If False, the images are shown.
    :param save_path: Path where the images should be saved. If None, a default path will be used.
    :param same_axis_both_plots: Determines whether the same y-axis should be used for both error-plots
    :param latex: Specifies whether the output should be additionally exportet in a pdf format (Only used if save is True)

    """

    # Ensure consistent colors and markers for each method

    method_styles = {
        'Smolyak': {'color': 'blue', 'marker': 'o', 'label': 'Smolyak'},
        'Least_Squares Uniform': {'color': 'orange', 'marker': 's', 'label': 'Least Squares Uniform'},
        'Least_Squares Chebyshev Weight': {'color': 'green', 'marker': '^', 'label': 'Least Squares Chebyshev Weight'}
    }

    if path is None:
        path = os.path.join(folder_name, "results_numerical_experiments.csv")

    if save_path is None:
        save_path = os.path.join(folder_name, "figures", function_type.name, f'dim{dimension}')

    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(path, sep=',', header=0)

    filtered_data = data[(data['dim'] == dimension) & (data['f_name'] == function_type.name)].copy()

    filtered_data.drop(['datetime', 'needed_time', 'sum_c', 'f_name'], axis=1, inplace=True)

    smolyak_data = filtered_data[(filtered_data['method']) == 'Smolyak']
    least_squares_data = filtered_data[(filtered_data['method']) == 'Least_Squares']
    boolean_series = (filtered_data['grid_type'] == 'CHEBYSHEV').reindex(least_squares_data.index,
                                                                         fill_value=False)
    least_squares_data_chebyshev_weight = least_squares_data[boolean_series]
    boolean_series = (filtered_data['grid_type'] == 'UNIFORM').reindex(least_squares_data.index,
                                                                       fill_value=False)
    least_squares_data_uniform = least_squares_data[boolean_series]

    smolyak_data = smolyak_data.sort_values(by='scale')
    least_squares_data_chebyshev_weight = least_squares_data_chebyshev_weight.sort_values(by='scale')
    least_squares_data_uniform = least_squares_data_uniform.sort_values(by='scale')

    # titles = ['Max (Abs) Error', 'L2 Error']
    titles = ['$e_{\ell_\infty}(A_j,q,f)$', '$e_{\ell_2}(A_j,q,f)$']
    errors = ['ell_infty_error', 'ell_2_error']

    global_min_uniform, global_max_uniform = None, None
    global_min_l2, global_max_l2 = None, None

    if not smolyak_data.empty:
        grouped = smolyak_data.groupby('c')
        for name, group in grouped:
            w = group['w'].iloc[0]
            if np.isinf(group['ell_infty_error']).any() or np.isinf(group['ell_2_error']).any():
                print(f"Skipping plot for {function_type.name}, c={name} and dimension {dimension} "
                      f"due to infinity values in errors.")
                continue
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=same_axis_both_plots)
            for i, error in enumerate(errors):
                axs[i].yaxis.set_tick_params(labelleft=True)
                data_temp = smolyak_data[smolyak_data['c'] == name]
                n_points_sy = data_temp.loc[smolyak_data['seed'] == seed, 'n_samples']
                data_temp = least_squares_data_chebyshev_weight[least_squares_data_chebyshev_weight['c'] == name]
                n_points_ls = data_temp.loc[least_squares_data_chebyshev_weight['seed'] == seed, 'n_samples']
                xticklabels = [f"{scale}\n{n_points_sy.iloc[j]}\n{n_points_ls.iloc[j]}" for j, scale in
                               enumerate(scales)]

                # Plotting Smolyak data with specific markers and colors
                smolyak_data_filtered = smolyak_data[smolyak_data['c'] == name]
                smolyak_plot_data = smolyak_data_filtered[smolyak_data_filtered['seed'] == seed]
                axs[i].plot(scales, smolyak_plot_data[error], label=method_styles['Smolyak']['label'],
                            color=method_styles['Smolyak']['color'], marker=method_styles['Smolyak']['marker'])

                # Plotting Least Squares Uniform data with specific markers and colors
                least_squares_data_uniform_filtered = least_squares_data_uniform[
                    least_squares_data_uniform['c'] == name]
                least_squares_plot_data_uniform = least_squares_data_uniform_filtered[
                    least_squares_data_uniform_filtered['seed'] == seed]
                axs[i].plot(scales, least_squares_plot_data_uniform[error],
                            label=method_styles['Least_Squares Uniform']['label'],
                            color=method_styles['Least_Squares Uniform']['color'],
                            marker=method_styles['Least_Squares Uniform']['marker'])

                # Plotting Least Squares Chebyshev Weight data with specific markers and colors
                least_squares_data_chebyshev_weight_filtered = least_squares_data_chebyshev_weight[
                    least_squares_data_chebyshev_weight['c'] == name]
                least_squares_plot_data_chebyshev_weight = least_squares_data_chebyshev_weight_filtered[
                    least_squares_data_chebyshev_weight_filtered['seed'] == seed]
                axs[i].plot(scales, least_squares_plot_data_chebyshev_weight[error],
                            label=method_styles['Least_Squares Chebyshev Weight']['label'],
                            color=method_styles['Least_Squares Chebyshev Weight']['color'],
                            marker=method_styles['Least_Squares Chebyshev Weight']['marker'])

                axs[i].set_xticks(scales)
                axs[i].set_xticklabels(xticklabels)
                axs[i].set_title(titles[i])
                axs[i].set_xlabel('scale ($q-d$)\npoints Smolyak\npoints Least Squares')
                axs[i].set_yscale('log')
                axs[i].legend()

                if error == 'ell_infty_error':
                    current_min_uniform, current_max_uniform = axs[i].get_ylim()
                    if global_min_uniform is None or current_min_uniform < global_min_uniform:
                        global_min_uniform = current_min_uniform
                    if global_max_uniform is None or current_max_uniform > global_max_uniform:
                        global_max_uniform = current_max_uniform
                elif error == 'ell_2_error':
                    current_min_l2, current_max_l2 = axs[i].get_ylim()
                    if global_min_l2 is None or current_min_l2 < global_min_l2:
                        global_min_l2 = current_min_l2
                    if global_max_l2 is None or current_max_l2 > global_max_l2:
                        global_max_l2 = current_max_l2

            axs[0].set_ylabel('estimated uniform error')
            axs[1].set_ylabel('estimated mean squared error')

            avg_c = np.mean(np.fromstring(name[1:-1], dtype=float, sep=','))
            fig.suptitle(
                f'{function_type.name}; multiplier={multiplier_fun(1.0)}; dim={dimension}'
                f'; avg_c={avg_c}\nc={name}\nw={w}')
            plt.tight_layout()
            if save:
                filename = get_next_filename(save_path)
                img_path = os.path.join(save_path, filename)
                plt.savefig(img_path)
                if latex:
                    plt.savefig(img_path.replace(".png", ".pdf"), format="pdf")
            else:
                plt.show()
            plt.close()
    else:
        print("Chebyshev data is empty, this is deprecated")


if __name__ == '__main__':

    seed = 42

    results_path = None
    dim_list = [4]
    scale_list = [1, 2, 3, 4,5,6]
    multiplier_fun = lambda x: 2 * x
    function_types = [FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK,
                      FunctionType.GAUSSIAN, FunctionType.CONTINUOUS, FunctionType.DISCONTINUOUS,
                      FunctionType.G_FUNCTION, FunctionType.MOROKOFF_CALFISCH_1, FunctionType.MOROKOFF_CALFISCH_2,
                      FunctionType.ROOS_ARNOLD, FunctionType.BRATLEY, FunctionType.ZHOU]

    folder_name = os.path.join("..", "results", "31_03_2025_07_17_20")

    # save all images in results folder
    total_iterations = len(dim_list) * len(function_types)
    with tqdm(total=total_iterations, desc="Plotting the results") as pbar:
        for dim in dim_list:
            for fun_type in function_types:
                plot_errors(dim, seed, fun_type, scale_list, multiplier_fun, save=True, folder_name=folder_name,
                            same_axis_both_plots=True, latex=True)
                pbar.update(1)
