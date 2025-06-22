import os
from typing import Union

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def estimate_runtimes(path: str, ylim: Union[None, int], save: bool = False, logarithmic: bool = False,
                      scales: list = None, dims: list = None, output_path: str = None,
                      sparse_ticks_fixed_scale: bool = False) -> None:
    """
    Estimates and plots runtimes: once as Runtime vs Dimension for each scale,
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

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    if scales is None:
        scales = sorted(df['scale'].unique().tolist())
    if dims is None:
        dims = sorted(df['dim'].unique().tolist())

    # Prepare runtime dicts
    ls_runtimes_scale = {scale: dict() for scale in scales}
    smolyak_runtimes_scale = {scale: dict() for scale in scales}
    ls_runtimes_dim = {dim: dict() for dim in dims}
    smolyak_runtimes_dim = {dim: dict() for dim in dims}

    unique_dims = df['dim'].unique().tolist()

    for (dim_name, dim_df) in df.groupby('dim'):
        for (scale_name, scale_df) in dim_df.groupby('scale'):
            for (method_name, method_df) in scale_df.groupby('method'):
                for (datetime_name, datetime_df) in method_df.groupby('datetime'):
                    runtime = datetime_df['needed_time'].iloc[0]

                    if scale_name in scales:
                        if method_name == 'Smolyak':
                            smolyak_runtimes_scale[scale_name].setdefault(dim_name, []).append(runtime)
                        else:
                            ls_runtimes_scale[scale_name].setdefault(dim_name, []).append(runtime)

                    if dim_name in dims:
                        if method_name == 'Smolyak':
                            smolyak_runtimes_dim[dim_name].setdefault(scale_name, []).append(runtime)
                        else:
                            ls_runtimes_dim[dim_name].setdefault(scale_name, []).append(runtime)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    color_map = matplotlib.colormaps.get_cmap('tab10')
    scale_colors = {scale: color_map(i % 10) for i, scale in enumerate(scales)}

    ax = axes[0]
    for scale, runtimes in ls_runtimes_scale.items():
        if scale not in scales:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'LS scale {scale}', marker='o',
                linestyle='-', color=scale_colors[scale])
    for scale, runtimes in smolyak_runtimes_scale.items():
        if scale not in scales:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'SA scale {scale}', marker='x',
                linestyle='--', color=scale_colors[scale])

    xticklabels = [str(dim) for dim in unique_dims]
    if sparse_ticks_fixed_scale:
        tick_indices = [i for i, dim in enumerate(unique_dims) if i % 10 == 0]
        tick_indices.append(len(unique_dims) - 1)  # Ensure the last tick is included
        tick_dims = [unique_dims[i] for i in tick_indices]
        tick_labels = [xticklabels[i] for i in tick_indices]
    else:
        tick_dims = unique_dims
        tick_labels = xticklabels

    ax.set_xticks(tick_dims)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs Dimension')
    if logarithmic:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.grid(True)
    ax.legend(loc="upper left")

    dim_colors = {dim: color_map(i % 10) for i, dim in enumerate(dims)}

    ax = axes[1]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for dim, runtimes in ls_runtimes_dim.items():
        if dim not in dims:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'LS dim {dim}', marker='o',
                linestyle='-', color=dim_colors[dim])
    for dim, runtimes in smolyak_runtimes_dim.items():
        if dim not in dims:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'SA dim {dim}', marker='x',
                linestyle='--', color=dim_colors[dim])
    ax.set_xlabel('Scale')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs Scale')
    if logarithmic:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save:
        if output_path is None:
            output_path = 'runtime_estimation_combined_plot.pdf'
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    path = os.path.join("..", "results", "final_results", "low_dim", "results_numerical_experiments.csv")
    dims_to_plot = [2, 5, 10]
    scales_to_plot = [2, 4, 6]
    estimate_runtimes(path, save=True, ylim=None, logarithmic=True, scales=scales_to_plot, dims=dims_to_plot,
                      sparse_ticks_fixed_scale=False)
