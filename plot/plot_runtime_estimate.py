import os
from collections import defaultdict

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_runtimes(path: str) -> dict:
    """
    Collects the runtimes of the experiments from the CSV file at the given path for each algorithm.

    :param path: Path to the CSV file containing the results.
    :return: A dictionary containing the runtimes for each algorithm, organized by method, dimension, and scale.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    name_combination_methods = ['algorithm', 'abbr_algorithm', 'method', 'abbr_method', 'basis_name',
                                'abbr_basis_name',
                                'grid_name', 'abbr_grid_name']

    runtimes_for_algorithm = defaultdict(dict)

    for (name, method_df) in df.groupby(name_combination_methods):
        algo_name = name[0]
        abbr_algo_name = name[1]
        method = name[2]
        abbr_method = name[3]
        basis_name = name[4]
        abbr_basis_name = name[5]
        grid_name = name[6]
        abbr_grid_name = name[7]

        method_name = f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}'

        runtimes_for_algorithm[method_name] = defaultdict(dict)

        for (dim_name, dim_df) in method_df.groupby('dim'):

            runtimes_for_algorithm[method_name][dim_name] = defaultdict(dict)

            for (scale_name, scale_df) in dim_df.groupby('scale'):
                runtimes_for_algorithm[method_name][dim_name][scale_name] = scale_df['needed_time'].mean()

    return runtimes_for_algorithm


def plot_runtimes_fixed_dim(path: str, ylim: int, save: bool = False, logarithmic: bool = False, dims: list = None,
                            output_path: str = None) -> None:
    """
    Creates a plot of the estimated runtimes for each algorithm based on the CSV file at the given path.
    :param path: Path to the CSV file containing the results.
    :param ylim: The upper limit for the y-axis in the plot.
    :param save: If True, save the plot in the specified output path.
    :param logarithmic: If True, use a logarithmic scale for the y-axis.
    :param dims: List of dimensions to plot. If None, all dimensions will be plotted.
    :param output_path: Path to save the plot if `save` is True.

    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    data = pd.read_csv(path, header=0, sep=',', decimal='.')

    if dims is None:
        dims = data['dim'].unique().tolist()

    data = get_runtimes(path)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = matplotlib.colormaps.get_cmap('tab10')
    markers = ['o', 's', 'D', '^', 'v', 'x', 'p', '*', 'h', '+']
    dim_colors = {dim: color_map(i % 10) for i, dim in enumerate(sorted(dims))}

    for index, (method_name, method_dict) in enumerate(data.items()):
        for dim, runtimes in method_dict.items():
            if dim not in dims:
                continue
            values = runtimes
            ax.plot(values.keys(), values.values(), label=f'{method_name} dim {dim}',
                    marker=markers[index % len(markers)], color=dim_colors[dim], alpha=0.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    ax.set_xlabel('Scale', fontsize=18)
    if logarithmic:
        ax.set_yscale('log')
    ax.set_ylabel('Runtime (seconds)', fontsize=18)
    ax.set_title('Runtime vs. Scale', fontsize=18)
    ax.legend(loc="upper left", fontsize=15)
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    plt.tight_layout()
    if save:
        if output_path is None:
            output_path = 'runtime_estimation_plot_fixed_dim.pdf'
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)


def plot_runtimes_fixed_scale(path: str, ylim: int, save: bool = False, logarithmic: bool = False, scales: list = None,
                              output_path: str = None, sparse_ticks: bool = False) -> None:
    """
    Creates a plot of the estimated runtimes for each algorithm based on the CSV file at the given path.
    :param path: Path to the CSV file containing the results.
    :param ylim: The upper limit for the y-axis in the plot.
    :param save: If True, save the plot in the specified output path.
    :param logarithmic: If True, use a logarithmic scale for the y-axis.
    :param scales: List of dimensions to plot. If None, all dimensions will be plotted.
    :param output_path: Path to save the plot if `save` is True.
    :param sparse_ticks: If True, use sparse ticks on the x-axis.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    data = pd.read_csv(path, header=0, sep=',', decimal='.')

    if scales is None:
        scales = data['scale'].unique().tolist()

    unique_dims = data['dim'].unique().tolist()

    data = get_runtimes(path)

    # swap the keys to have scales as the first level
    result = dict()
    for h_key, g_dict in data.items():
        new_inner = defaultdict(dict)
        for g_key, f_dict in g_dict.items():
            for f_key, value in f_dict.items():
                new_inner[f_key][g_key] = value
        result[h_key] = dict(new_inner)
    data = result

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = matplotlib.colormaps.get_cmap('tab10')
    markers = ['o', 's', 'D', '^', 'v', 'x', 'p', '*', 'h', '+']
    scale_colors = {dim: color_map(i % 10) for i, dim in enumerate(sorted(scales))}

    for index, (method_name, method_dict) in enumerate(data.items()):
        for scale, runtimes in method_dict.items():
            if scale not in scales:
                continue
            values = runtimes
            ax.plot(values.keys(), values.values(), label=f'{method_name} scale {scale}',
                    marker=markers[index % len(markers)], color=scale_colors[scale], alpha=0.5)

    xticklabels = [str(dim) for dim in unique_dims]
    if sparse_ticks:
        tick_indices = [i for i, dim in enumerate(unique_dims) if i % 10 == 0]
        tick_indices.append(len(unique_dims) - 1)  # Ensure the last tick is included
        tick_dims = [unique_dims[i] for i in tick_indices]
        tick_labels = [xticklabels[i] for i in tick_indices]
    else:
        tick_dims = unique_dims
        tick_labels = xticklabels
    ax.set_xticks(tick_dims)
    ax.set_xticklabels(tick_labels, fontsize=15)

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    ax.set_xlabel('Dimension', fontsize=18)
    if logarithmic:
        ax.set_yscale('log')
    ax.set_ylabel('Runtime (seconds)', fontsize=18)
    ax.set_title('Runtime vs. Dimension', fontsize=18)
    ax.legend(loc="upper left", fontsize=15)
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    plt.tight_layout()
    if save:
        if output_path is None:
            output_path = 'runtime_estimation_plot_fixed_scale.pdf'
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)


def plot_runtime_side_by_side(path: str, ylim: int, save: bool = False, logarithmic: bool = False, scales: list = None,
                              dims: list = None, output_path: str = None,
                              sparse_ticks_fixed_scale: bool = False) -> None:
    """
        Plots runtimes: once as Runtime vs Dimension for each scale,
        and once as Runtime vs Scale for each dimension.

        :param path: Path to CSV file.
        :param ylim: Upper limit for y-axis.
        :param save: If True, save the plot to output_path.
        :param logarithmic: If True, use log scale for y-axis.
        :param scales: List of scales to plot (for left plot).
        :param dims: List of dimensions to plot (for right plot).
        :param output_path: Path to save the figure if `save` is True.
        :param sparse_ticks_fixed_scale: If True, use sparse ticks for fixed scale plots.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    data = pd.read_csv(path, header=0, sep=',', decimal='.')

    if scales is None:
        scales = sorted(data['scale'].unique().tolist())
    if dims is None:
        dims = sorted(data['dim'].unique().tolist())

    unique_dims = data['dim'].unique().tolist()

    data = get_runtimes(path)

    swapped_data = dict()
    for h_key, g_dict in data.items():
        new_inner = defaultdict(dict)
        for g_key, f_dict in g_dict.items():
            for f_key, value in f_dict.items():
                new_inner[f_key][g_key] = value
        swapped_data[h_key] = dict(new_inner)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    color_map = matplotlib.colormaps.get_cmap('tab10')

    scale_colors = {scale: color_map(i % 10) for i, scale in enumerate(scales)}
    dim_colors = {dim: color_map(i % 10) for i, dim in enumerate(dims)}

    markers = ['o', 's', 'D', '^', 'v', 'x', 'p', '*', 'h', '+']

    # Left plot:
    ax = axes[0]
    for index, (method_name, method_dict) in enumerate(swapped_data.items()):
        for scale, runtimes in method_dict.items():
            if scale not in scales:
                continue
            values = runtimes
            ax.plot(values.keys(), values.values(), label=f'{method_name} scale {scale}',
                    marker=markers[index % len(markers)], color=scale_colors[scale], alpha=0.5)

    xticklabels = [str(dim) for dim in unique_dims]
    if sparse_ticks_fixed_scale:
        tick_indices = [i for i, dim in enumerate(unique_dims) if i % 10 == 0]
        tick_indices.append(len(unique_dims) - 1)  # Ensure the last tick is included
        tick_dims = [unique_dims[i] for i in tick_indices]
        tick_labels = [xticklabels[i] for i in tick_indices]
    else:
        tick_dims = unique_dims
        tick_labels = xticklabels

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    ax.set_xticks(tick_dims)
    ax.set_xticklabels(tick_labels, fontsize=15)
    ax.set_xlabel('Dimension', fontsize=18)
    ax.set_ylabel('Runtime (seconds)', fontsize=18)
    ax.set_title('Runtime vs. Dimension', fontsize=18)
    if logarithmic:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=15)

    # Right plot:
    ax = axes[1]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for index, (method_name, method_dict) in enumerate(data.items()):
        for dim, runtimes in method_dict.items():
            if dim not in dims:
                continue
            values = runtimes
            ax.plot(values.keys(), values.values(), label=f'{method_name} dim {dim}',
                    marker=markers[index % len(markers)], color=dim_colors[dim], alpha=0.5)

    ax.set_xlabel('Scale', fontsize=18)
    ax.set_ylabel('Runtime (seconds)', fontsize=18)
    ax.set_title('Runtime vs. Scale', fontsize=18)
    if logarithmic:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=15)

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    for label in ax.get_xticklabels():
        label.set_fontsize(15)

    plt.tight_layout()

    if save:
        if output_path is None:
            output_path = 'runtime_estimation_combined_plot.pdf'
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    path = os.path.join("..", "results", "18_07_2025_17_27_13", "results_numerical_experiments.csv")

    dims = [2, 4, 5]
    scales = [2, 4, 6]
    plot_runtime_side_by_side(path, ylim=None, save=False, logarithmic=True, scales=scales, dims=dims,
                              sparse_ticks_fixed_scale=False)
