import os
import time

import pandas as pd
import matplotlib.pyplot as plt


def estimate_runtimes(path: str, save: bool = False, output_path: str = None) -> None:
    """
    Estimates the runtimes of the experiments from the CSV file at the given path for each algorithm.

    :param path: Path to the CSV file containing the results.
    :param save: If True, save the plot in the specified output path.
    :param output_path: Path to save the plot if `save` is True.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    # for each dim and algorithm, we want to plot the runtime for different scales as a function runtime vs dim

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
                    if method_name == 'Smolyak':
                        # ignore all entries that have method_type != 'TASMANIAN'
                        if datetime_df['method_type'].iloc[0] != 'TASMANIAN':
                            continue

                        runtime = datetime_df['needed_time'].iloc[0]
                        smolyak_runtimes[scale_name][dim_name] = runtime
                    else:
                        runtime = datetime_df['needed_time'].iloc[0]
                        ls_runtimes[scale_name][dim_name] = runtime

    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))

    # x-axis is scale, y-axis is runtime, each algorithm is plotted for each dim
    for scale, runtimes in ls_runtimes.items():
        ax.plot(runtimes.keys(), runtimes.values(), label=f'LS {scale}', marker='o')
    for scale, runtimes in smolyak_runtimes.items():
        ax.plot(runtimes.keys(), runtimes.values(), label=f'Smolyak {scale}', marker='x')
    ax.set_xlabel('Dimension')
    # ax.set_yscale('log')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime of Algorithms vs Dimension')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        if output_path is None:
            output_path = 'runtime_estimation_plot.png'
        plt.savefig(output_path)
    else:
        plt.show()

    time.sleep(100)
    plt.close(fig)


if __name__ == '__main__':
    path = os.path.join("..", "results", "final_results", "results_numerical_experiments.csv")
    estimate_runtimes(path, save = False)
