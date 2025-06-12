import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.cm as cm

def poly_fit(x, a, b, c):
    return a * x ** 2 + b * x + c

def exp_fit(x, a, b):
    return a * np.exp(b * x)

def estimate_runtimes_by_scale(path: str, ylim: int, extrapolate_to: int, save: bool = False, output_path: str = None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path, header=0, sep=',', decimal='.')

    ls_runtimes = dict()
    smolyak_runtimes = dict()

    for (dim_name, dim_df) in df.groupby('dim'):
        dim = int(dim_name)
        if dim not in ls_runtimes:
            ls_runtimes[dim] = dict()
        if dim not in smolyak_runtimes:
            smolyak_runtimes[dim] = dict()

        for (scale_name, scale_df) in dim_df.groupby('scale'):
            scale = int(scale_name)
            for (method_name, method_df) in scale_df.groupby('method'):
                for (_, datetime_df) in method_df.groupby('datetime'):
                    runtime = datetime_df['needed_time'].iloc[0]
                    if method_name == 'Smolyak':
                        if datetime_df['method_type'].iloc[0] != 'TASMANIAN':
                            continue
                        smolyak_runtimes[dim][scale] = runtime
                    else:
                        ls_runtimes[dim][scale] = runtime

    fig, ax = plt.subplots(figsize=(10, 6))

    def plot_smoothed(x, y, label, color, linestyle, ylim: int, extrapolate_to: int):
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        x_vals = np.array(x_sorted)
        y_vals = np.array(y_sorted)

        try:
            popt, _ = curve_fit(exp_fit, x_vals, y_vals, maxfev=10000)
            x_range = np.linspace(min(x_vals), extrapolate_to, 200)
            y_fit = exp_fit(x_range, *popt)
            fit_type = "exp"
        except RuntimeError:
            try:
                popt, _ = curve_fit(poly_fit, x_vals, y_vals)
                x_range = np.linspace(min(x_vals), extrapolate_to, 200)
                y_fit = poly_fit(x_range, *popt)
                fit_type = "poly"
            except RuntimeError:
                ax.plot(x_vals, y_vals, marker='o', label=f'{label} (raw)', color=color, linestyle=linestyle)
                return

        y_fit_masked = np.ma.masked_where((y_fit < 0) | (y_fit > ylim), y_fit)
        ax.plot(x_range, y_fit_masked, label=f'{label} ({fit_type})', color=color, linestyle=linestyle)

    unique_dims = sorted(set(ls_runtimes.keys()) | set(smolyak_runtimes.keys()))
    color_map = {dim: cm.tab10(i % 10) for i, dim in enumerate(unique_dims)}

    for dim in unique_dims:
        color = color_map[dim]
        if dim in ls_runtimes:
            x = list(ls_runtimes[dim].keys())
            y = list(ls_runtimes[dim].values())
            plot_smoothed(x, y, label=f'LS dim {dim}', color=color, linestyle='-', ylim=ylim, extrapolate_to=extrapolate_to)
        if dim in smolyak_runtimes:
            x = list(smolyak_runtimes[dim].keys())
            y = list(smolyak_runtimes[dim].values())
            plot_smoothed(x, y, label=f'SA dim {dim}', color=color, linestyle='--', ylim=ylim, extrapolate_to=extrapolate_to)

    ax.set_xlabel('Scale')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_ylim(0, ylim)
    ax.set_title('Smoothed Runtime Estimation by Scale per Dimension')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save:
        if output_path is None:
            output_path = 'runtime_estimation_by_scale.png'
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    path = os.path.join("..", "results", "final_results", "low_dim", "results_numerical_experiments.csv")
    estimate_runtimes_by_scale(path, save=False, ylim=10, extrapolate_to=10)
