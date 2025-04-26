import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_all_errors_fixed_dim(file_name: str, plot_type: str = "boxplot", box_plot_width: float = 0.15,
                              save: bool = False, latex: bool = False, only_maximum: bool = False):
    """
        Creates distribution plots for each function class at a certain dimension
        The ell2 and the max error are plotted.

        :param file_name: csv-filename in which the results are stored
        :param plot_type: Either boxplot or errorbar
        :param box_plot_width: width of the boxplots that are drawn
        :param save: Specifies whether the images should be saved. If False, the images are shown.
        :param latex: Specifies whether the output should be additionally exported in a pdf format (Only used if save is True)
        :param only_maximum: If True, only the maximum error is plotted
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

    total_plots = len(function_types) * len(dimensions)

    with tqdm(total=total_plots, desc="Plotting errors") as pbar:
        for f_type in function_types:
            data_f_type = df[df['f_name'] == f_type].copy()
            for dim in dimensions:
                data_dim = data_f_type[data_f_type['dim'] == dim].copy()
                # Create the figure with two subplots
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
                axs[1].yaxis.set_tick_params(labelleft=True)
                scales = sorted(data_dim['scale'].unique())

                n_points_ls = []
                n_points_sy = []

                n_functions_list = []

                for index, grid in enumerate(grid_types):

                    data_grid_type = data_dim[data_dim['grid_type'] == str(grid)].copy()

                    if grid == "SPARSE":
                        method = "Smolyak"
                    elif grid == "UNIFORM" or grid == "CHEBYSHEV":
                        method = "Least_Squares"
                    else:
                        raise ValueError(f"Cannot handle the selected grid_type {grid}")

                    # Drop unnecessary columns
                    data_grid_type.drop(['datetime', 'needed_time', 'sum_c', 'f_name', 'c', 'w', 'seed'], axis=1,
                                        inplace=True,
                                        errors='ignore')

                    # Define an offset based on the index
                    offset = (index - len(
                        grid_types) / 2) * box_plot_width * 0.95  # Spread out boxplots slightly

                    # Get a color and marker for the current method-grid combination
                    c = colors[index % len(colors)]
                    marker = markers[index % len(markers)]

                    mean_values_ellinf = []
                    mean_values_ell2 = []

                    max_values_ellinf = []
                    max_values_ell2 = []

                    for scale in scales:
                        scale_data = data_grid_type[data_grid_type['scale'] == scale].copy()

                        n_functions_list.append(len(scale_data))

                        if grid == "SPARSE":
                            n_points_sy.append(scale_data['n_samples'].iloc[0])
                        elif grid == "UNIFORM":
                            n_points_ls.append(scale_data['n_samples'].iloc[0])

                        # Compute means
                        mean_ellinf = scale_data['ell_infty_error'].mean()
                        mean_ell2 = scale_data['ell_2_error'].mean()

                        mean_values_ellinf.append(mean_ellinf)
                        mean_values_ell2.append(mean_ell2)

                        if only_maximum:

                            max_values_ellinf.append(scale_data['ell_infty_error'].max())
                            max_values_ell2.append(scale_data['ell_2_error'].max())

                        else:
                            if plot_type == "boxplot":
                                # Boxplots
                                axs[0].boxplot(scale_data['ell_2_error'], positions=[scale + offset], showfliers=False,
                                               widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                               whis=[0, 100],
                                               whiskerprops=dict(color=c), capprops=dict(color=c),
                                               medianprops=dict(color=c))
                                axs[1].boxplot(scale_data['ell_infty_error'], positions=[scale + offset],
                                               showfliers=False,
                                               widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                               whis=[0, 100],
                                               whiskerprops=dict(color=c), capprops=dict(color=c),
                                               medianprops=dict(color=c))

                            elif plot_type == "errorbar":
                                max_ellinf = scale_data['ell_infty_error'].max()
                                max_ell2 = scale_data['ell_2_error'].max()

                                # Error bars
                                axs[0].errorbar(scale, mean_ell2, yerr=[[0], [max_ell2 - mean_ell2]],
                                                fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                                elinewidth=1.5)
                                axs[1].errorbar(scale, mean_ellinf, yerr=[[0], [max_ellinf - mean_ellinf]],
                                                fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                                elinewidth=1.5)

                    if only_maximum:
                        axs[0].plot(scales, max_values_ell2, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')
                        axs[1].plot(scales, max_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')


                    else:
                        axs[0].plot(scales, mean_values_ell2, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')
                        axs[1].plot(scales, mean_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')

                    # get the number of points in smolyak for each scale uniquely

                pbar.update(1)

                xticklabels = [f"{scale}\n{n_points_sy[j]}\n{n_points_ls[j]}" for j, scale in enumerate(scales)]

                for ax in axs:
                    ax.set_xlabel('scale ($=q-d$)\npoints Smolyak\npoints Least Squares')
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(False)
                    ax.set_xticks(scales)  # Ensure ticks correspond to original scales
                    ax.set_xticklabels(xticklabels)  # Explicitly label them as integers

                if not only_maximum:
                    axs[0].set_ylabel('estimated mean squared error')
                    axs[1].set_ylabel('estimated uniform error')
                else:
                    axs[0].set_ylabel('estimated max mean squared error')
                    axs[1].set_ylabel('estimated max uniform error')

                # Adjust the layout and show the plot
                plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
                plt.figtext(0.06, 0.95, f"$Q={min(n_functions_list)}$", fontsize=8, verticalalignment='top',
                            horizontalalignment='left', color='gray')

                if save:
                    if only_maximum:
                        save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'dim{dim}',
                                                 'max_error_distribution_fixed_dim.png')
                    else:
                        save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'dim{dim}',
                                                 'error_distribution_fixed_dim.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    if latex:
                        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
                else:
                    plt.show()
                plt.close()


def plot_all_errors_fixed_scale(file_name: str, plot_type: str = "boxplot", box_plot_width: float = 0.15,
                                save: bool = False, latex: bool = False, only_maximum: bool = False):
    """
        Creates distribution plots for each function class at a certain scale
        The ell2 and the max error are plotted.

        :param file_name: csv-filename in which the results are stored
        :param plot_type: Either boxplot or errorbar
        :param box_plot_width: width of the boxplots that are drawn
        :param save: Specifies whether the images should be saved. If False, the images are shown.
        :param latex: Specifies whether the output should be additionally exportet in a pdf format (Only used if save is True)
        :param only_maximum: If True, only the maximum error is plotted
    """

    if plot_type not in ["boxplot", "errorbar"]:
        raise ValueError(f"The plotting-type {plot_type} is not supported! Use 'boxplot' or 'errorbar'!")

    df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']  # Different markers

    # Get distinct values for dimension, function type, grids, methods, and scales
    scales = df['scale'].unique()
    function_types = df['f_name'].unique()
    grid_types = df['grid_type'].unique()

    total_plots = len(function_types) * len(scales)

    with tqdm(total=total_plots, desc="Plotting errors") as pbar:
        for f_type in function_types:
            data_f_type = df[df['f_name'] == f_type].copy()
            for scale in scales:
                data_scale = data_f_type[data_f_type['scale'] == scale].copy()
                # Create the figure with two subplots
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
                axs[1].yaxis.set_tick_params(labelleft=True)
                dims = sorted(data_scale['dim'].unique())

                n_points_ls = []
                n_points_sy = []

                n_functions_list = []

                for index, grid in enumerate(grid_types):

                    data_grid_type = data_scale[data_scale['grid_type'] == str(grid)].copy()

                    if grid == "SPARSE":
                        method = "Smolyak"
                    elif grid == "UNIFORM" or grid == "CHEBYSHEV":
                        method = "Least_Squares"
                    else:
                        raise ValueError(f"Cannot handle the selected grid_type {grid}")

                    # Drop unnecessary columns
                    data_grid_type.drop(['datetime', 'needed_time', 'sum_c', 'f_name', 'c', 'w', 'seed'], axis=1,
                                        inplace=True,
                                        errors='ignore')

                    # Define an offset based on the index
                    offset = (index - len(
                        grid_types) / 2) * box_plot_width * 0.95  # Spread out boxplots slightly

                    # Get a color and marker for the current method-grid combination
                    c = colors[index % len(colors)]
                    marker = markers[index % len(markers)]

                    mean_values_ellinf = []
                    mean_values_ell2 = []

                    max_values_ellinf = []
                    max_values_ell2 = []

                    for dim in dims:
                        dim_data = data_grid_type[data_grid_type['dim'] == dim].copy()

                        n_functions_list.append(len(dim_data))

                        if grid == "SPARSE":
                            n_points_sy.append(dim_data['n_samples'].iloc[0])
                        elif grid == "UNIFORM":
                            n_points_ls.append(dim_data['n_samples'].iloc[0])

                        # Compute means
                        mean_ellinf = dim_data['ell_infty_error'].mean()
                        mean_ell2 = dim_data['ell_2_error'].mean()

                        mean_values_ellinf.append(mean_ellinf)
                        mean_values_ell2.append(mean_ell2)

                        if only_maximum:
                            max_values_ellinf.append(dim_data['ell_infty_error'].max())
                            max_values_ell2.append(dim_data['ell_2_error'].max())

                        else:
                            if plot_type == "boxplot":
                                # Boxplots
                                axs[0].boxplot(dim_data['ell_2_error'], positions=[dim + offset], showfliers=False,
                                               widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                               whiskerprops=dict(color=c), capprops=dict(color=c), whis=[0, 100],
                                               medianprops=dict(color=c))
                                axs[1].boxplot(dim_data['ell_infty_error'], positions=[dim + offset], showfliers=False,
                                               widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                               whiskerprops=dict(color=c), capprops=dict(color=c), whis=[0, 100],
                                               medianprops=dict(color=c))



                            elif plot_type == "errorbar":
                                max_ellinf = dim_data['ell_infty_error'].max()
                                max_ell2 = dim_data['ell_2_error'].max()

                                # Error bars
                                axs[0].errorbar(dim, mean_ell2, yerr=[[0], [max_ell2 - mean_ell2]],
                                                fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                                elinewidth=1.5)
                                axs[1].errorbar(dim, mean_ellinf, yerr=[[0], [max_ellinf - mean_ellinf]],
                                                fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                                elinewidth=1.5)

                    if only_maximum:
                        axs[0].plot(dims, max_values_ell2, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')

                        axs[1].plot(dims, max_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')


                    else:
                        axs[0].plot(dims, mean_values_ell2, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')
                        axs[1].plot(dims, mean_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker,
                                    linestyle='-')

                pbar.update(1)

                xticklabels = [f"{dim}\n{n_points_sy[j]}\n{n_points_ls[j]}" for j, dim in enumerate(dims)]

                for ax in axs:
                    ax.set_xlabel('dim \npoints Smolyak\npoints Least Squares')
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(False)
                    ax.set_xticks(dims)  # Ensure ticks correspond to original scales
                    ax.set_xticklabels(xticklabels)  # Explicitly label them as integers

                if not only_maximum:
                    axs[0].set_ylabel('estimated mean squared error')
                    axs[1].set_ylabel('estimated uniform error')
                else:
                    axs[0].set_ylabel('estimated max mean squared error')
                    axs[1].set_ylabel('estimated max uniform error')

                # Adjust the layout and show the plot
                plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
                plt.figtext(0.06, 0.95, f"$Q\geq {min(n_functions_list)}$", fontsize=8, verticalalignment='top',
                            horizontalalignment='left', color='gray')

                if save:
                    if only_maximum:
                        save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'scale{scale}',
                                                 'max_error_distribution_fixed_scale.png')
                    else:
                        save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'scale{scale}',
                                                 'error_distribution_fixed_scale.png')

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    if latex:
                        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
                else:
                    plt.show()
                plt.close()


if __name__ == '__main__':
    plottype = "boxplot"

    folder_name = os.path.join("..", "results")
    filename = "path/to/your/results_numerical_experiments.csv"  # os.path.join(folder_name, "combined_results_numerical_experiments.csv")

    plot_all_errors_fixed_dim(filename, save=True, latex=True, plot_type=plottype, only_maximum=False)
    plot_all_errors_fixed_scale(filename, save=True, latex=True, plot_type=plottype, only_maximum=False)
    plot_all_errors_fixed_dim(filename, save=True, latex=True, plot_type=plottype, only_maximum=True)
    plot_all_errors_fixed_scale(filename, save=True, latex=True, plot_type=plottype, only_maximum=True)
