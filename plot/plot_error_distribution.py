"""
Error distribution plots: boxplots / error bars of ℓ₂ and ℓ∞ errors
across scales (fixed dim) or across dimensions (fixed scale).
"""

from collections import defaultdict

import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_all_errors_fixed_dim(file_name: str = None, df: pd.DataFrame = None,
                              dim: int = None, save_dir: str = None,
                              abbreviation_dict: dict = None, plot_type: str = "boxplot",
                              box_plot_width: float = 0.15,
                              save: bool = False, latex: bool = False, only_maximum: bool = False,
                              skip_scale_one_distribution: bool = True, verbose: bool = False):
    """
        Creates distribution plots for each function class at a certain dimension
        The ell2 and the max error are plotted.

        :param file_name: csv-filename in which the results are stored (ignored when *df* is given)
        :param df: In-memory DataFrame with results (preferred over reading CSV)
        :param dim: Dimension for which the plots should be created (ignored when None)
        :param save_dir: Directory for saved figures.  When *None*, derived from *file_name*.
        :param abbreviation_dict: Dictionary to abbreviate function names in the plots
        :param plot_type: Either boxplot or errorbar
        :param box_plot_width: width of the boxplots that are drawn
        :param save: Specifies whether the images should be saved. If False, the images are shown.
        :param latex: Specifies whether the output should be additionally exported in a pdf format (Only used if save is True)
        :param only_maximum: If True, only the maximum error is plotted
        :param skip_scale_one_distribution: If True, the first boxplot is skipped.
        :param verbose: If True, a progress bar is shown for the plotting process.
    """

    if plot_type not in ["boxplot", "errorbar"]:
        raise ValueError(f"The plotting-type {plot_type} is not supported! Use 'boxplot' or 'errorbar'!")

    if abbreviation_dict is None:
        abbreviation_dict = {
            "CONTINUOUS": "Continuous",
            "CORNER_PEAK": "Corner Peak",
            "DISCONTINUOUS": "Discontinuous",
            "G_FUNCTION": "Ridge Product",
            "GAUSSIAN": "Gaussian",
            "MOROKOFF_CALFISCH_1": "Geometric Mean",
            "OSCILLATORY": "Oscillatory",
            "PRODUCT_PEAK": "Product Peak",
            "ZHOU": "Bimodal Gaussian",
            "NOISE": "Noise"
        }

    if df is None:
        if file_name is None:
            raise ValueError("Either file_name or df must be provided")
        df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    if save_dir is None and file_name is not None:
        save_dir = os.path.dirname(file_name)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']

    # Get distinct values for dimension, function type, grids, methods, and scales
    if dim is None:
        dimensions = df['dim'].unique()
    else:
        dimensions = [dim]
    function_types = df['f_name'].unique()

    total_plots = len(function_types) * len(dimensions)

    if verbose:
        iterator = tqdm([(f, d) for f in function_types for d in dimensions],
                        desc="Plotting errors")
    else:
        iterator = [(f, d) for f in function_types for d in dimensions]


    for f_type, dim in iterator:
        data_f_type = df[df['f_name'] == f_type].copy()
        data_dim = data_f_type[data_f_type['dim'] == dim].copy()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        axs[1].yaxis.set_tick_params(labelleft=True)
        scales = sorted(data_dim['scale'].unique())

        n_points_train = defaultdict(list)
        n_points_test = defaultdict(list)

        n_functions_list = []

        name_combination_methods = ['algorithm', 'abbr_algorithm', 'method', 'abbr_method', 'basis_name',
                                    'abbr_basis_name',
                                    'grid_name', 'abbr_grid_name']

        grouped = data_dim.groupby(name_combination_methods)

        n_groups = grouped.ngroups

        for index, (name, group) in enumerate(grouped):
            algo_name = name[0]
            abbr_algo_name = name[1]
            method = name[2]
            abbr_method = name[3]
            basis_name = name[4]
            abbr_basis_name = name[5]
            grid_name = name[6]
            abbr_grid_name = name[7]

            offset = ((index - n_groups) / 2) * box_plot_width * 1.1

            c = colors[index % len(colors)]
            marker = markers[index % len(markers)]

            mean_values_ellinf = []
            mean_values_ell2 = []

            max_values_ellinf = []
            max_values_ell2 = []

            for scale in scales:
                scale_data = group[group['scale'] == scale].copy()

                n_functions_list.append(len(scale_data))

                n_points_train[scale].append(scale_data['n_samples'].iloc[0])
                n_points_test[scale].append(scale_data['n_test_samples'].iloc[0])

                mean_ellinf = scale_data['ell_infty_error'].mean()
                mean_ell2 = scale_data['ell_2_error'].mean()

                mean_values_ellinf.append(mean_ellinf)
                mean_values_ell2.append(mean_ell2)

                if only_maximum:

                    max_values_ellinf.append(scale_data['ell_infty_error'].max())
                    max_values_ell2.append(scale_data['ell_2_error'].max())

                else:
                    if plot_type == "boxplot":
                        if not (skip_scale_one_distribution and scale == 1):
                            axs[0].boxplot(scale_data['ell_infty_error'], positions=[scale + offset],
                                           showfliers=False,
                                           widths=box_plot_width,
                                           boxprops=dict(color=c, linestyle='--'), whis=[0, 100],
                                           whiskerprops=dict(color=c), capprops=dict(color=c),
                                           medianprops=dict(color=c))
                            axs[1].boxplot(scale_data['ell_2_error'], positions=[scale + offset],
                                           showfliers=False, widths=box_plot_width,
                                           boxprops=dict(color=c, linestyle='--'), whis=[0, 100],
                                           whiskerprops=dict(color=c), capprops=dict(color=c),
                                           medianprops=dict(color=c))

                    elif plot_type == "errorbar":
                        max_ellinf = scale_data['ell_infty_error'].max()
                        max_ell2 = scale_data['ell_2_error'].max()

                        if not (skip_scale_one_distribution and scale == 1):
                            axs[0].errorbar(scale, mean_ellinf, yerr=[[0], [max_ellinf - mean_ellinf]],
                                            fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7,
                                            ecolor=c, elinewidth=1.5)
                            axs[1].errorbar(scale, mean_ell2, yerr=[[0], [max_ell2 - mean_ell2]],
                                            fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7,
                                            ecolor=c, elinewidth=1.5)

            if only_maximum:
                axs[0].plot(scales, max_values_ellinf,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)
                axs[1].plot(scales, max_values_ell2,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)


            else:
                axs[0].plot(scales, mean_values_ellinf,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)
                axs[1].plot(scales, mean_values_ell2,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)

        xticklabels = [f"{scale}\n{min(n_points_train[scale])}\n{min(n_points_test[scale])}" for scale in
                       scales]
        axs[0].set_xlabel('scale\nmin points train\nmin points test', fontsize=14, linespacing=1.2)

        for ax in axs:
            ax.xaxis.set_label_coords(1.1875, -0.025)
            ax.set_yscale('log')
            ax.legend(fontsize=14)
            ax.grid(False)
            ax.set_xticks(scales)
            ax.set_xticklabels(xticklabels)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)

        if not only_maximum:
            axs[0].set_ylabel(r'$e_{\mathrm{max}}$', fontsize=18)
            axs[1].set_ylabel(r'$e_{\mathrm{mean}}$', fontsize=18)
        else:
            axs[0].set_ylabel(r'$e_{\mathrm{max}}^{\mathrm{wc}}$', fontsize=18)
            axs[1].set_ylabel(r'$e_{\mathrm{mean}}^{\mathrm{wc}}$', fontsize=18)

        plt.tight_layout(rect=(0.00, 0.00, 1.0, 0.95))
        plt.subplots_adjust(wspace=0.35)

        fig.suptitle(f"{abbreviation_dict[f_type]}, $d={dim}$, $Q={min(n_functions_list)}$", fontsize=16,
                     fontweight='bold', x=0.525)

        if save:
            if only_maximum:
                save_path = os.path.join(save_dir, "figures", f_type, "dim",
                                         f'dim{dim}_max_error_distribution_fixed_dim.png')
            else:
                save_path = os.path.join(save_dir, "figures", f_type, "dim",
                                         f'dim{dim}_error_distribution_fixed_dim.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            if latex:
                plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
        else:
            plt.show()
        plt.close()


def plot_all_errors_fixed_scale(file_name: str = None, df: pd.DataFrame = None,
                                scale: int = None, save_dir: str = None,
                                abbreviation_dict: dict = None, plot_type: str = "boxplot",
                                box_plot_width: float = 0.15,
                                save: bool = False, latex: bool = False, only_maximum: bool = False,
                                sparse_ticks: bool = False, verbose: bool = False):
    """
        Creates distribution plots for each function class at a certain scale
        The ell2 and the max error are plotted.

        :param file_name: csv-filename in which the results are stored (ignored when *df* is given)
        :param df: In-memory DataFrame with results (preferred over reading CSV)
        :param scale: Scale for which the plots should be created (ignored when None)
        :param save_dir: Directory for saved figures.  When *None*, derived from *file_name*.
        :param abbreviation_dict: Dictionary to abbreviate function names in the plots
        :param plot_type: Either boxplot or errorbar
        :param box_plot_width: width of the boxplots that are drawn
        :param save: Specifies whether the images should be saved. If False, the images are shown.
        :param latex: Specifies whether the output should be additionally exported in a pdf format (Only used if save is True)
        :param only_maximum: If True, only the maximum error is plotted
        :param sparse_ticks: If True, the x-ticks are only shown for some dimensions
        :param verbose: If True, a progress bar is shown for the plotting process.
    """

    if plot_type not in ["boxplot", "errorbar"]:
        raise ValueError(f"The plotting-type {plot_type} is not supported! Use 'boxplot' or 'errorbar'!")

    if abbreviation_dict is None:
        abbreviation_dict = {
            "CONTINUOUS": "Continuous",
            "CORNER_PEAK": "Corner Peak",
            "DISCONTINUOUS": "Discontinuous",
            "G_FUNCTION": "Ridge Product",
            "GAUSSIAN": "Gaussian",
            "MOROKOFF_CALFISCH_1": "Geometric Mean",
            "OSCILLATORY": "Oscillatory",
            "PRODUCT_PEAK": "Product Peak",
            "ZHOU": "Bimodal Gaussian",
            "NOISE": "Noise"
        }

    if df is None:
        if file_name is None:
            raise ValueError("Either file_name or df must be provided")
        df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    if save_dir is None and file_name is not None:
        save_dir = os.path.dirname(file_name)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']

    # Get distinct values for dimension, function type, grids, methods, and scales
    if scale is None:
        scales = df['scale'].unique()
    else:
        scales = [scale]
    function_types = df['f_name'].unique()

    total_plots = len(function_types) * len(scales)

    if verbose:
        iterator = tqdm([(f, s) for f in function_types for s in scales],
                        desc="Plotting errors")
    else:
        iterator = [(f, s) for f in function_types for s in scales]


    for f_type, scale in iterator:
        data_f_type = df[df['f_name'] == f_type].copy()
        data_scale = data_f_type[data_f_type['scale'] == scale].copy()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        axs[1].yaxis.set_tick_params(labelleft=True)
        dims = sorted(data_scale['dim'].unique())

        n_points_train = defaultdict(list)
        n_points_test = defaultdict(list)

        n_functions_list = []

        name_combination_methods = ['algorithm', 'abbr_algorithm', 'method', 'abbr_method', 'basis_name',
                                    'abbr_basis_name',
                                    'grid_name', 'abbr_grid_name']

        grouped = data_scale.groupby(name_combination_methods)

        n_groups = grouped.ngroups

        for index, (name, group) in enumerate(grouped):
            algo_name = name[0]
            abbr_algo_name = name[1]
            method = name[2]
            abbr_method = name[3]
            basis_name = name[4]
            abbr_basis_name = name[5]
            grid_name = name[6]
            abbr_grid_name = name[7]

            offset = ((index - n_groups) / 2) * box_plot_width * 1.1

            c = colors[index % len(colors)]
            marker = markers[index % len(markers)]

            mean_values_ellinf = []
            mean_values_ell2 = []

            max_values_ellinf = []
            max_values_ell2 = []

            for dim in dims:
                dim_data = group[group['dim'] == dim].copy()

                n_functions_list.append(len(dim_data))

                n_points_train[dim].append(dim_data['n_samples'].iloc[0])
                n_points_test[dim].append(dim_data['n_test_samples'].iloc[0])

                mean_ellinf = dim_data['ell_infty_error'].mean()
                mean_ell2 = dim_data['ell_2_error'].mean()

                mean_values_ellinf.append(mean_ellinf)
                mean_values_ell2.append(mean_ell2)

                if only_maximum:
                    max_values_ellinf.append(dim_data['ell_infty_error'].max())
                    max_values_ell2.append(dim_data['ell_2_error'].max())

                else:
                    if plot_type == "boxplot":
                        axs[0].boxplot(dim_data['ell_infty_error'], positions=[dim + offset], showfliers=False,
                                       widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                       whiskerprops=dict(color=c), capprops=dict(color=c), whis=[0, 100],
                                       medianprops=dict(color=c))
                        axs[1].boxplot(dim_data['ell_2_error'], positions=[dim + offset], showfliers=False,
                                       widths=box_plot_width, boxprops=dict(color=c, linestyle='--'),
                                       whiskerprops=dict(color=c), capprops=dict(color=c), whis=[0, 100],
                                       medianprops=dict(color=c))

                    elif plot_type == "errorbar":
                        max_ellinf = dim_data['ell_infty_error'].max()
                        max_ell2 = dim_data['ell_2_error'].max()

                        axs[0].errorbar(dim, mean_ellinf, yerr=[[0], [max_ellinf - mean_ellinf]],
                                        fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                        elinewidth=1.5)
                        axs[1].errorbar(dim, mean_ell2, yerr=[[0], [max_ell2 - mean_ell2]],
                                        fmt=marker, color=c, capsize=5, linestyle='None', alpha=0.7, ecolor=c,
                                        elinewidth=1.5)

            if only_maximum:
                axs[0].plot(dims, max_values_ellinf,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)
                axs[1].plot(dims, max_values_ell2,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)

            else:
                axs[0].plot(dims, mean_values_ellinf,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)
                axs[1].plot(dims, mean_values_ell2,
                            label=f'{abbr_algo_name}-{abbr_method}-{abbr_basis_name}-{abbr_grid_name}', color=c,
                            marker=marker,
                            linestyle='-', alpha=0.7)

        xticklabels = [f"{dim}\n{min(n_points_train[dim])}\n{min(n_points_test[dim])}" for dim in dims]
        axs[0].set_xlabel(f'$d$\nmin points train\nmin points test', fontsize=12, linespacing=1.05)

        for ax in axs:

            ax.xaxis.set_label_coords(1.175, -0.025)
            ax.set_yscale('log')
            ax.legend(fontsize=14)
            ax.grid(False)

            if sparse_ticks:
                tick_indices = [i for i, dim in enumerate(dims) if i % 10 == 0]
                tick_indices.append(len(dims) - 1)  # Ensure the last tick is included
                tick_dims = [dims[i] for i in tick_indices]
                tick_labels = [xticklabels[i] for i in tick_indices]
            else:
                tick_dims = dims
                tick_labels = xticklabels

            ax.set_xticks(tick_dims)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=15)

        if not only_maximum:
            axs[0].set_ylabel(r'$e_{\mathrm{max}}$', fontsize=18)
            axs[1].set_ylabel(r'$e_{\mathrm{mean}}$', fontsize=18)
        else:
            axs[0].set_ylabel(r'$e_{\mathrm{max}}^{\mathrm{wc}}$', fontsize=18)
            axs[1].set_ylabel(r'$e_{\mathrm{mean}}^{\mathrm{wc}}$', fontsize=18)

        plt.tight_layout(rect=(0.00, 0.00, 1.0, 0.95))
        plt.subplots_adjust(wspace=0.35)

        fig.suptitle(f"{abbreviation_dict[f_type]}, $scale={scale}$, $Q={min(n_functions_list)}$", fontsize=16,
                     fontweight='bold', x=0.525)

        if save:
            if only_maximum:
                save_path = os.path.join(save_dir, "figures", f_type, "scale",
                                         f'scale{scale}_max_error_distribution_fixed_scale.png')
            else:
                save_path = os.path.join(save_dir, "figures", f_type, "scale",
                                         f'scale{scale}_error_distribution_fixed_scale.png')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            if latex:
                plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    filename = "path/to/your/results_numerical_experiments.csv"

    plot_all_errors_fixed_dim(filename, save=True, latex=True, plot_type="boxplot", only_maximum=True)
    plot_all_errors_fixed_dim(filename, save=True, latex=True, plot_type="boxplot", only_maximum=False)
