import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_all_errors(file_name: str, plot_type: str = "errorbar", box_plot_width: float = 0.15, save: bool = False,
                    latex: bool = False):
    """
        Creates distribution plots for each function class at a certain dimension and scale
        The ell2 and the max error are plotted.

        :param file_name: csv-filename in which the results are stored
        :plot_type: Either boxplot or errorbar
        :param box_plot_width: width of the boxplots that are drawn
        :param save: Specifies whether the images should be saved. If False, the images are shown.
        :param latex: Specifies whether the output should be additionally exportet in a pdf format (Only used if save is True)
    """

    if plot_type not in ["boxplot", "errorbar"]:
        raise ValueError(f"The plotting-type {plot_type} is not supported! Use 'boxplot' or 'errorbar'!")

    df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']  # Different markers

    # Get distinct values for dimension, function type, grids, methods, and scales
    dimensions = df['dim'].unique()
    function_types = df['f_name'].unique()
    grid_types = df['grid_type'].unique()
    scales = sorted(df['scale'].unique())

    total_plots = len(function_types) * len(dimensions)

    with tqdm(total=total_plots, desc="Plotting errors") as pbar:
        for f_type in function_types:
            for dim in dimensions:
                # Create the figure with two subplots
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
                axs[1].yaxis.set_tick_params(labelleft=True)

                for index, grid in enumerate(grid_types):
                    # Filter the relevant data
                    data = df[
                        (df['dim'] == dim) &
                        (df['f_name'] == f_type) &
                        (df['grid_type'] == str(grid))
                        ].copy()

                    if grid == "SPARSE":
                        method = "Smolyak"
                    elif grid == "UNIFORM" or grid == "CHEBYSHEV":
                        method = "Least_Sqares"
                    else:
                        raise ValueError(f"Cannot handle the selected grid_type {grid}")

                    # Drop unnecessary columns
                    data.drop(['datetime', 'needed_time', 'sum_c', 'f_name', 'c', 'w'], axis=1, inplace=True,
                              errors='ignore')

                    # Define an offset based on the index
                    offset = (index - len(
                        grid_types) / 2) * box_plot_width * 0.95  # Spread out boxplots slightly

                    # Get a color and marker for the current method-grid combination
                    c = colors[index % len(colors)]
                    marker = markers[index % len(markers)]

                    mean_values_ellinf = []
                    mean_values_ell2 = []
                    n_points_ls = []
                    n_points_sy = []

                    for scale in scales:
                        scale_data = data[data['scale'] == scale]

                        # get number of points
                        n_points_sy.append(scale_data['n_samples'].iloc[0])
                        n_points_ls.append(scale_data['n_samples'].iloc[0])

                        # Compute means
                        mean_ellinf = scale_data['ell_infty_error'].mean()
                        mean_ell2 = scale_data['ell_2_error'].mean()

                        mean_values_ellinf.append(mean_ellinf)
                        mean_values_ell2.append(mean_ell2)

                        if plot_type == "boxplot":
                            # Boxplots
                            axs[0].boxplot(scale_data['ell_infty_error'], positions=[scale + offset], showfliers=False,
                                           widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                           whiskerprops=dict(color=c), capprops=dict(color=c),
                                           medianprops=dict(color='black'))

                            axs[1].boxplot(scale_data['ell_2_error'], positions=[scale + offset], showfliers=False,
                                           widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                           whiskerprops=dict(color=c), capprops=dict(color=c),
                                           medianprops=dict(color='black'))

                        elif plot_type == "errorbar":
                            max_ellinf = scale_data['ell_infty_error'].max()
                            max_ell2 = scale_data['ell_2_error'].max()

                            # Error bars
                            axs[0].errorbar(scale, mean_ellinf, yerr=[[0], [max_ellinf - mean_ellinf]],
                                            fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                            elinewidth=1.5)
                            axs[1].errorbar(scale, mean_ell2, yerr=[[0], [max_ell2 - mean_ell2]],
                                            fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                            elinewidth=1.5)

                    axs[0].plot(scales, mean_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker,
                                linestyle='-')
                    axs[1].plot(scales, mean_values_ell2, label=f'{method} - {grid}', color=c, marker=marker,
                                linestyle='-')

                    # get the number of points in smolyak for each scale uniquely

                pbar.update(1)

                xticklabels = [f"{scale}\n{n_points_sy[j]}\n{n_points_ls[j]}" for j, scale in enumerate(scales)]

                for ax in axs:
                    ax.set_xlabel('scale ($q-d$)\npoints Smolyak\npoints Least Squares')
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(False)
                    ax.set_xticks(scales)  # Ensure ticks correspond to original scales
                    ax.set_xticklabels(xticklabels)  # Explicitly label them as integers

                axs[0].set_ylabel('estimated uniform error')
                axs[1].set_ylabel('estimated mean squared error')

                # Adjust layout and show the plot
                plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

                if save:
                    save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'dim{dim}',
                                             'error_distribution.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    if latex:
                        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
                else:
                    plt.show()
                plt.close()


if __name__ == '__main__':
    plottype = "boxplot"

    folder_name = os.path.join("..", "results", "1_1_1")
    filename = os.path.join(folder_name, "results_numerical_experiments.csv")

    plot_all_errors(filename, save=True, latex=True, plot_type=plottype)
