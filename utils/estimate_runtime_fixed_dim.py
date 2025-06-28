import os
from typing import Union

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def estimate_runtimes(path: str, ylim: Union[None, int], save: bool = False, logarithmic: bool = False,
                      dims: list = None,
                      output_path: str = None) -> None:
    """
    Estimates the runtimes of the experiments from the CSV file at the given path for each algorithm.

    :param path: Path to the CSV file containing the results.
    :param ylim: The upper limit for the y-axis in the plot.
    :param save: If True, save the plot in the specified output path.
    :param logarithmic: If True, use a logarithmic scale for the y-axis.
    :param dims: List of dimensions to plot. If None, all dimensions will be plotted.
    :param output_path: Path to save the plot if `save` is True.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    if dims is None:
        dims = df['dim'].unique().tolist()

    ls_runtimes = dict()
    smolyak_runtimes = dict()

    for (scale_name, scale_df) in df.groupby('scale'):
        for (dim_name, dim_df) in scale_df.groupby('dim'):
            if dim_name not in ls_runtimes:
                ls_runtimes[dim_name] = dict()
            if dim_name not in smolyak_runtimes:
                smolyak_runtimes[dim_name] = dict()
            for (method_name, method_df) in dim_df.groupby('method'):
                for (datetime_name, datetime_df) in method_df.groupby('datetime'):
                    runtime = datetime_df['needed_time'].iloc[0]
                    if method_name == 'Smolyak':
                        if scale_name not in smolyak_runtimes[dim_name]:
                            smolyak_runtimes[dim_name][scale_name] = [runtime]
                        else:
                            smolyak_runtimes[dim_name][scale_name].append(runtime)
                    else:
                        if scale_name not in ls_runtimes[dim_name]:
                            ls_runtimes[dim_name][scale_name] = [runtime]
                        else:
                            ls_runtimes[dim_name][scale_name].append(runtime)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = matplotlib.colormaps.get_cmap('tab10')
    dim_colors = {dim: color_map(i % 10) for i, dim in enumerate(sorted(dims))}

    for dim, runtimes in ls_runtimes.items():
        if dim not in dims:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'LS dim {dim}', marker='o',
                linestyle='-', color=dim_colors[dim])

    for dim, runtimes in smolyak_runtimes.items():
        if dim not in dims:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'SA dim {dim}', marker='x',
                linestyle='--', color=dim_colors[dim])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Scale')
    if logarithmic:
        ax.set_yscale('log')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs. Scale')
    ax.legend(loc="upper left")
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    plt.tight_layout()
    if save:
        if output_path is None:
            output_path = 'runtime_estimation_plot.pdf'
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)
