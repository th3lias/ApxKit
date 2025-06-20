import os
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator


def estimate_runtimes(path: str, ylim: Union[None, int], save: bool = False, logarithmic: bool = False,
                      scales: list = None,
                      output_path: str = None) -> None:
    """
    Estimates the runtimes of the experiments from the CSV file at the given path for each algorithm.

    :param path: Path to the CSV file containing the results.
    :param ylim: The upper limit for the y-axis in the plot.
    :param save: If True, save the plot in the specified output path.
    :param logarithmic: If True, use a logarithmic scale for the y-axis.
    :param scales: List of scales to plot. If None, all scales will be plotted.
    :param output_path: Path to save the plot if `save` is True.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    if scales is None:
        scales = df['scale'].unique().tolist()

    ls_runtimes = dict()
    smolyak_runtimes = dict()

    for (dim_name, dim_df) in df.groupby('dim'):
        for (scale_name, scale_df) in dim_df.groupby('scale'):
            if scale_name not in ls_runtimes:
                ls_runtimes[scale_name] = dict()
            if scale_name not in smolyak_runtimes:
                smolyak_runtimes[scale_name] = dict()
            for (method_name, method_df) in scale_df.groupby('method'):
                for (datetime_name, datetime_df) in method_df.groupby('datetime'):
                    runtime = datetime_df['needed_time'].iloc[0]
                    if method_name == 'Smolyak':
                        if dim_name not in smolyak_runtimes[scale_name]:
                            smolyak_runtimes[scale_name][dim_name] = [runtime]
                        else:
                            smolyak_runtimes[scale_name][dim_name].append(runtime)
                    else:
                        if dim_name not in ls_runtimes[scale_name]:
                            ls_runtimes[scale_name][dim_name] = [runtime]
                        else:
                            ls_runtimes[scale_name][dim_name].append(runtime)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = matplotlib.colormaps.get_cmap('tab10')
    scale_colors = {scale: color_map(i % 10) for i, scale in enumerate(sorted(scales))}

    for scale, runtimes in ls_runtimes.items():
        if scale not in scales:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'LS scale {scale}', marker='o',
                linestyle='-', color=scale_colors[scale])

    for scale, runtimes in smolyak_runtimes.items():
        if scale not in scales:
            continue
        values = {k: sum(v) / len(v) for k, v in runtimes.items()}
        ax.plot(values.keys(), values.values(), label=f'SA scale {scale}', marker='x',
                linestyle='--', color=scale_colors[scale])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Dimension')
    if logarithmic:
        ax.set_yscale('log')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime of Algorithms vs Dimension')
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


if __name__ == '__main__':
    path = os.path.join("..", "results", "final_results", "low_dim", "results_numerical_experiments.csv")
    scales_to_plot = [2, 4, 6]
    estimate_runtimes(path, save=False, ylim=None, logarithmic=True, scales=scales_to_plot)
