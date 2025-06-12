import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit


def poly_fit(x, a, b, c):
    return a * x ** 2 + b * x + c


def exp_fit(x, a, b):
    return a * np.exp(b * x)


def estimate_runtimes_2(path: str, ylim: int, extrapolate_to: int, save: bool = False, output_path: str = None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

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
                        if datetime_df['method_type'].iloc[0] != 'TASMANIAN':
                            continue
                        runtime = datetime_df['needed_time'].iloc[0]
                        smolyak_runtimes[scale_name][int(dim_name)] = runtime
                    else:
                        runtime = datetime_df['needed_time'].iloc[0]
                        ls_runtimes[scale_name][int(dim_name)] = runtime

    fig, ax = plt.subplots(figsize=(10, 6))

    def plot_smoothed(ax, x, y, label, color, linestyle, ylim: int, extrapolate_to=20):
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        x_vals = np.array(x_sorted)
        y_vals = np.array(y_sorted)

        try:
            popt, _ = curve_fit(exp_fit, x_vals, y_vals, maxfev=10000)
            x_range = np.linspace(1, extrapolate_to, 200)
            y_fit = exp_fit(x_range, *popt)
        except RuntimeError:
            try:
                popt, _ = curve_fit(poly_fit, x_vals, y_vals)
                x_range = np.linspace(1, extrapolate_to, 200)
                y_fit = poly_fit(x_range, *popt)
            except RuntimeError:
                ax.plot(x_vals, y_vals, marker='o', label=label, color=color, linestyle=linestyle)
                return

        y_fit_masked = np.ma.masked_where((y_fit < 0) | (y_fit > ylim), y_fit)
        ax.plot(x_range, y_fit_masked, label=label, color=color, linestyle=linestyle)

    unique_scales = sorted(set(ls_runtimes.keys()) | set(smolyak_runtimes.keys()))
    color_map = {scale: cm.tab10(i % 10) for i, scale in enumerate(unique_scales)}

    for scale in unique_scales:
        color = color_map[scale]
        if scale in ls_runtimes:
            x = list(ls_runtimes[scale].keys())
            y = list(ls_runtimes[scale].values())
            plot_smoothed(ax, x, y, label=f'LS scale {scale}', color=color, linestyle='-', ylim=ylim,
                          extrapolate_to=extrapolate_to)
        if scale in smolyak_runtimes:
            x = list(smolyak_runtimes[scale].keys())
            y = list(smolyak_runtimes[scale].values())
            plot_smoothed(ax, x, y, label=f'SA scale {scale}', color=color, linestyle='--', ylim=ylim,
                          extrapolate_to=extrapolate_to)

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_ylim(0, ylim)
    ax.set_title('Smoothed Runtime Estimation of Algorithms')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        if output_path is None:
            output_path = 'runtime_estimation_plot_smoothed.png'
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    path = os.path.join("..", "results", "final_results", "low_dim", "results_numerical_experiments.csv")
    estimate_runtimes_2(path, save=False, ylim=10, extrapolate_to=10)
