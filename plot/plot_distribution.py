import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_all_errors(file_name: str, box_plot_width: float = 0.15, save: bool = False):
    df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']  # Different markers

    df['method_grid_combo'] = df['method'] + '|' + df['grid_type']

    # Get distinct values for dimension, function type, grids, methods, and scales
    dimensions = df['dim'].unique()
    function_types = df['f_name'].unique()
    different_combis = df['method_grid_combo'].unique()
    scales = sorted(df['scale'].unique())

    total_plots = len(function_types) * len(dimensions)

    with tqdm(total=total_plots, desc="Plotting errors") as pbar:
        for f_type in function_types:
            for dim in dimensions:
                # Create the figure with two subplots
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
                axs[1].yaxis.set_tick_params(labelleft=True)
                fig.suptitle(f'{f_type} - Dimension{dim}', fontsize=16)

                for index, elem in enumerate(different_combis):
                    method, grid = elem.split('|')
                    # Filter the relevant data
                    data = df[
                        (df['dim'] == dim) &
                        (df['f_name'] == f_type) &
                        (df['grid_type'] == str(grid)) &
                        (df['method'] == str(method))
                        ].copy()

                    # Drop unnecessary columns
                    data.drop(['datetime', 'needed_time', 'sum_c', 'f_name', 'c', 'w'], axis=1, inplace=True,
                              errors='ignore')

                    # Define an offset based on the index
                    offset = (index - len(different_combis) / 2) * box_plot_width * 0.9  # Spread out boxplots slightly

                    # Get a color and marker for the current method-grid combination
                    c = colors[index % len(colors)]
                    marker = markers[index % len(markers)]

                    # Prepare data for `ell_infty_error`
                    mean_values_ellinf = []
                    for scale in scales:
                        scale_data = data[data['scale'] == scale]
                        mean_values_ellinf.append(scale_data['ell_infty_error'].mean())
                    axs[0].plot(scales, mean_values_ellinf, label=f'{method} - {grid}', color=c, marker=marker)
                    for scale in scales:
                        scale_data = data[data['scale'] == scale]['ell_infty_error']
                        axs[0].boxplot(scale_data,
                                       positions=[scale + offset],  # Use offset for plotting
                                       showfliers=False,
                                       widths=box_plot_width,
                                       boxprops=dict(color=c, linestyle='--'),  # Dashed outline for box
                                       whiskerprops=dict(color=c),  # Set whisker color
                                       capprops=dict(color=c),  # Set cap color
                                       medianprops=dict(color='black'))  # Set median line color

                    # Prepare data for `ell_2_error`
                    mean_values_ell2 = []
                    for scale in scales:
                        scale_data = data[data['scale'] == scale]
                        mean_values_ell2.append(scale_data['ell_2_error'].mean())
                    axs[1].plot(scales, mean_values_ell2, label=f'{method} - {grid}', color=c, marker=marker)
                    for scale in scales:
                        scale_data = data[data['scale'] == scale]['ell_2_error']
                        axs[1].boxplot(scale_data,
                                       positions=[scale + offset],  # Use offset for plotting
                                       showfliers=False,
                                       widths=box_plot_width,
                                       boxprops=dict(color=c, linestyle='--'),  # Dashed outline for box
                                       whiskerprops=dict(color=c),  # Set whisker color
                                       capprops=dict(color=c),  # Set cap color
                                       medianprops=dict(color='black'))  # Set median line color

                pbar.update(1)
                # Formatting the left subplot (ell_infty_error)
                axs[0].set_title('ell_infty_error')
                axs[0].set_xlabel('Scale')
                axs[0].set_ylabel('Error')
                axs[0].set_yscale('log')
                axs[0].legend()
                axs[0].grid(False)
                axs[0].set_xticks(scales)  # Ensure ticks correspond to original scales
                axs[0].set_xticklabels([f'{s}' for s in scales])  # Explicitly label them as integers

                # Formatting the right subplot (ell_2_error)
                axs[1].set_title('ell_2_error')
                axs[1].set_xlabel('Scale')
                axs[1].set_yscale('log')
                axs[1].legend()
                axs[1].grid(False)
                axs[1].set_xticks(scales)  # Ensure ticks correspond to original scales
                axs[1].set_xticklabels([f'{s}' for s in scales])  # Explicitly label them as integers

                # Adjust layout and show the plot
                plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

                if save:
                    save_path = os.path.join(os.path.dirname(file_name), "figures", f_type, f'dim{dim}',
                                             'error_distribution.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    plt.close()
                else:
                    plt.show()
                    plt.close()
