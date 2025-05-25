import datetime
import pandas as pd
import numpy as np
import os

from typing import List, Union, Dict


def highlight_matching_value(value, min_value):
    return r"\first{" + f"{value:.2e}" + r"}" if np.isclose(value, min_value, atol=1e-17) else f"{value:.2e}"


def generate_table_fixed_dim(results_csv_path: str, output_folder: str, skip_mean_error: bool = False,
                             skip_scale: Union[Dict, None] = None):
    r""" Creates a tex file for each dimension in the specified csv file. The tex files will be stored in the given folder.

        In the LaTeX File, the table can be printed via
        \begin{table}[htbp]
            \label{tab:dim4_results}
            \centering
            \begin{adjustbox}{width=\linewidth}
                \input{>>TEX-FILE-PATH<<}
            \end{adjustbox}
            \vspace{0.1cm}
            \caption{Test\label{tab:dim4_results}}
        \end{table}

        For that all the following packages are needed
        % Packages for the tabulars
        \usepackage{booktabs}
        \usepackage{array}
        \usepackage{adjustbox}
        \usepackage{multirow}
        \usepackage{makecell}

        Additionally, we need the following commands
        % Command for the first entry in a row
        \newcommand{\first}[1]{\textbf{#1}}
    """

    abbreviation_dict = {
        "BRATLEY": "Bratley",
        "CONTINUOUS": "Continuous",
        "CORNER_PEAK": "Corner Peak",
        "DISCONTINUOUS": "Discontinuous",
        "G_FUNCTION": "Modified Ridge Product",
        "GAUSSIAN": "Gaussian",
        "MOROKOFF_CALFISCH_1": "Modified Geometric Mean",
        "MOROKOFF_CALFISCH_2": "Morokoff Calfisch 2",
        "OSCILLATORY": "Oscillatory",
        "PRODUCT_PEAK": "Product Peak",
        "ROOS_ARNOLD": "Roos Arnold",
        "ZHOU": "Bimodal Gaussian"
    }

    output = dict()

    error_reductions = ['max']

    if not skip_mean_error:
        error_reductions.insert(0, 'mean')
        errors = [r'e_{\rm mean}', r'e_{\rm max}']
    else:
        errors = [r'e_{\rm mean}^{\rm wc}', r'e_{\rm max}^{\rm wc}']

    no_error_combinations = len(errors) * len(error_reductions)

    results = pd.read_csv(results_csv_path, sep=',', header=0, decimal='.')

    for dim_name, dim_df in results.groupby('dim'):

        if skip_scale is None:
            skip_scale = dict()

        skip_scale_dim = skip_scale.get(str(dim_name), [])

        # get scales
        scales = sorted(dim_df['scale'].unique())
        scales = [s for s in scales if s not in skip_scale_dim]

        right_text = ("|" + ("r" * no_error_combinations)) * len(scales)

        output[dim_name] = f"% Created with Python on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "\n"
        output[dim_name] += f"% {results_csv_path}, dim={dim_name}, scales = {[int(s) for s in scales]}" + "\n"
        output[dim_name] += r"\begin{tabular}{ll" + right_text + r"|}" + "\n"

        # add header
        output[dim_name] += r" &  "
        for i, scale in enumerate(scales):
            if i == 0:
                output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                    no_error_combinations) + r"}{c}{Scale" + str(scale) + r"}"
            else:
                output[dim_name] += r" & \multicolumn{" + str(no_error_combinations) + r"}{c}{Scale" + str(scale) + r"}"
        output[dim_name] += r"\\" + "\n"

        if len(error_reductions) > 1:
            output[dim_name] += r" &  "
            for i, scale in enumerate(scales):
                for j, error_reduction in enumerate(error_reductions):
                    if i == 0 and j == 0:
                        output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                            len(errors)) + r"}{c}{" + str(error_reduction) + r"}"
                    else:
                        output[dim_name] += r" & \multicolumn{" + str(len(errors)) + r"}{c}{" + str(
                            error_reduction) + r"}"
            output[dim_name] += r"\\" + "\n"

        output[dim_name] += r" &  "
        for i, scale in enumerate(scales):
            for j, error_reduction in enumerate(error_reductions):
                for k, error in enumerate(errors):
                    if i == 0 and j == 0 and k == 0:
                        output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{1}{c}{$" + error + r"$}"
                    else:
                        output[dim_name] += r" & \multicolumn{1}{c}{$" + error + r"$}"
        output[dim_name] += r"\\" + "\n" + r"\toprule" + "\n"

        dfg = dim_df.groupby('f_name')
        for fun_index, (fun_name, fun_df) in enumerate(dfg):
            min_values = {}

            n_functions_list = []

            for i, scale in enumerate(scales):
                scale_df = fun_df[fun_df['scale'] == scale]

                ell_2_max = []
                ell_infty_max = []

                if not skip_mean_error:
                    ell_2_mean = []
                    ell_infty_mean = []

                for method_name, method_df in scale_df.groupby('grid_type', sort=False):

                    n_functions_list.append(len(method_df))

                    ell_2_max.append(method_df['ell_2_error'].max())
                    ell_infty_max.append(method_df['ell_infty_error'].max())

                    if not skip_mean_error:
                        ell_2_mean.append(method_df['ell_2_error'].mean())
                        ell_infty_mean.append(method_df['ell_infty_error'].mean())

                min_values[str(scale)] = {
                    'ell_2_max': min(ell_2_max),
                    'ell_infty_max': min(ell_infty_max),
                }

                if not skip_mean_error:
                    min_values[str(scale)] = {
                        'ell_2_max': min(ell_2_max),
                        'ell_infty_max': min(ell_infty_max),
                        'ell_2_mean': min(ell_2_mean),
                        'ell_infty_mean': min(ell_infty_mean),
                    }

            for grid_index, (grid_name, grid_df) in enumerate(fun_df.groupby('grid_type', sort=False)):

                # filter out the scales that are not in the scales list
                grid_df = grid_df[grid_df['scale'].isin(scales)]

                for scale_index, (scale_name, scale_df) in enumerate(grid_df.groupby('scale')):

                    if grid_name == 'CHEBYSHEV':
                        method_name = "LS-Chebyshev"
                    elif grid_name == 'UNIFORM':
                        method_name = "LS-Uniform"
                    elif grid_name == 'SPARSE':
                        method_name = 'Smolyak'
                    else:
                        raise ValueError(f"Can't handle grid: {grid_name}")

                    if scale_index == 0:
                        if grid_index == 0:
                            output[dim_name] += r"\multirow{3}{*}{\thead[l]{\tiny\textbf{" + abbreviation_dict[
                                str(fun_name)] + r"}\\" + r"$Q=" + str(min(n_functions_list)) + r"$}} & "
                        else:
                            output[dim_name] += r" & "
                        output[dim_name] += method_name
                    else:
                        output[dim_name] += r" "

                    ell_2_error = scale_df['ell_2_error']
                    ell_infty_error = scale_df['ell_infty_error']

                    ell_infty_max = ell_infty_error.max()
                    ell_2_max = ell_2_error.max()

                    if not skip_mean_error:
                        ell_infty_mean = ell_infty_error.mean()
                        ell_2_mean = ell_2_error.mean()

                    if skip_mean_error:
                        output[dim_name] += (
                                r" & "
                                + highlight_matching_value(ell_2_max, min_values[str(scale_name)]['ell_2_max'])
                                + r" & "
                                + highlight_matching_value(ell_infty_max, min_values[str(scale_name)]['ell_infty_max'])
                        )
                    else:
                        output[dim_name] += (
                                r" & "
                                + highlight_matching_value(ell_2_mean, min_values[str(scale_name)]['ell_2_mean'])
                                + r" & "
                                + highlight_matching_value(ell_infty_mean,
                                                           min_values[str(scale_name)]['ell_infty_mean'])
                                + r" & "
                                + highlight_matching_value(ell_2_max, min_values[str(scale_name)]['ell_2_max'])
                                + r" & "
                                + highlight_matching_value(ell_infty_max, min_values[str(scale_name)]['ell_infty_max'])
                        )

                output[dim_name] += r"\\" + "\n"
            if not fun_index == len(dfg) - 1:
                output[dim_name] += r"\midrule" + "\n"
        output[dim_name] += r"\bottomrule" + "\n"
        output[dim_name] += r"\end{tabular}" + "\n"

    for dim_name, table in output.items():
        with open(os.path.join(output_folder, f"dim{dim_name}.tex"), "w") as f:
            f.write(table)
        print("Exported table for dim", dim_name, "to", os.path.join(output_folder, f"dim{dim_name}.tex"))


def generate_table_fixed_scale(results_csv_path: str, output_folder: str, skip_mean_error: bool = False,
                               skip_dim: Union[Dict, None] = None):
    r""" Creates a tex file for each scale in the specified csv file. The tex files will be stored in the given folder.

        In the LaTeX File, the table can be printed via
        \begin{table}[htbp]
            \label{tab:scale1_results}
            \centering
            \begin{adjustbox}{width=\linewidth}
                \input{>>TEX-FILE-PATH<<}
            \end{adjustbox}
            \vspace{0.1cm}
            \caption{Test\label{tab:scale1_results}}
        \end{table}
    """

    abbreviation_dict = {
        "BRATLEY": "Bratley",
        "CONTINUOUS": "Continuous",
        "CORNER_PEAK": "Corner Peak",
        "DISCONTINUOUS": "Discontinuous",
        "G_FUNCTION": "Modified Ridge Product",
        "GAUSSIAN": "Gaussian",
        "MOROKOFF_CALFISCH_1": "Modified Geometric Mean",
        "MOROKOFF_CALFISCH_2": "Morokoff Calfisch 2",
        "OSCILLATORY": "Oscillatory",
        "PRODUCT_PEAK": "Product Peak",
        "ROOS_ARNOLD": "Roos Arnold",
        "ZHOU": "Bimodal Gaussian"
    }

    output = dict()

    error_reductions = ['max']

    if not skip_mean_error:
        error_reductions.insert(0, 'mean')
        errors = [r'e_{\rm mean}', r'e_{\rm max}']
    else:
        errors = [r'e_{\rm mean}^{\rm wc}', r'e_{\rm max}^{\rm wc}']

    no_error_combinations = len(errors) * len(error_reductions)
    results = pd.read_csv(results_csv_path, sep=',', header=0, decimal='.')

    for scale_name, scale_df in results.groupby('scale'):

        if skip_dim is None:
            skip_dim = dict()

        skip_dims_for_scale = skip_dim.get(str(scale_name), [])

        dims = sorted(scale_df['dim'].unique())
        dims = [d for d in dims if d not in skip_dims_for_scale]

        right_text = ("|" + ("r" * no_error_combinations)) * len(dims)

        output[scale_name] = f"% Created with Python on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output[scale_name] += f"% {results_csv_path}, scale={scale_name}, dims = {[int(d) for d in dims]}\n"
        output[scale_name] += r"\begin{tabular}{ll" + right_text + r"|}" + "\n"

        # Header rows
        output[scale_name] += r" &  "
        for i, dim in enumerate(dims):
            if i == 0:
                output[scale_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                    no_error_combinations) + r"}{c}{Dim" + str(dim) + r"}"
            else:
                output[scale_name] += r" & \multicolumn{" + str(no_error_combinations) + r"}{c}{Dim" + str(dim) + r"}"
        output[scale_name] += r"\\" + "\n"

        if len(error_reductions) > 1:
            output[scale_name] += r" &  "
            for i in range(len(dims)):
                for j, error_reduction in enumerate(error_reductions):
                    if i == 0 and j == 0:
                        output[scale_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                            len(errors)) + r"}{c}{" + str(
                            error_reduction) + r"}"
                    else:
                        output[scale_name] += r" & \multicolumn{" + str(len(errors)) + r"}{c}{" + str(
                            error_reduction) + r"}"
            output[scale_name] += r"\\" + "\n"

        output[scale_name] += r" &  "
        for i in range(len(dims)):
            for j in range(len(error_reductions)):
                for k, error in enumerate(errors):
                    if i == 0 and j == 0 and k == 0:
                        output[scale_name] += r" \multicolumn{1}{c}{} & \multicolumn{1}{c}{$" + error + r"$}"
                    else:
                        output[scale_name] += r" & \multicolumn{1}{c}{$" + error + r"$}"
        output[scale_name] += r"\\" + "\n" + r"\toprule" + "\n"

        dfg = scale_df.groupby('f_name')
        for fun_index, (fun_name, fun_df) in enumerate(dfg):
            min_values = {}

            n_functions_list = []

            for dim in dims:
                dim_df = fun_df[fun_df['dim'] == dim]

                ell_2_max = []
                ell_infty_max = []
                if not skip_mean_error:
                    ell_2_mean = []
                    ell_infty_mean = []

                for method_name, method_df in dim_df.groupby('grid_type', sort=False):

                    n_functions_list.append(len(method_df))

                    ell_2_max.append(method_df['ell_2_error'].max())
                    ell_infty_max.append(method_df['ell_infty_error'].max())
                    if not skip_mean_error:
                        ell_2_mean.append(method_df['ell_2_error'].mean())
                        ell_infty_mean.append(method_df['ell_infty_error'].mean())

                min_values[str(dim)] = {
                    'ell_2_max': min(ell_2_max),
                    'ell_infty_max': min(ell_infty_max),
                }

                if not skip_mean_error:
                    min_values[str(dim)].update({
                        'ell_2_mean': min(ell_2_mean),
                        'ell_infty_mean': min(ell_infty_mean),
                    })

            for grid_index, (grid_name, grid_df) in enumerate(fun_df.groupby('grid_type', sort=False)):

                grid_df = grid_df[grid_df['dim'].isin(dims)]

                for dim_index, (dim, dim_df) in enumerate(grid_df.groupby('dim')):

                    if grid_name == 'CHEBYSHEV':
                        method_name = "LS-Chebyshev"
                    elif grid_name == 'UNIFORM':
                        method_name = "LS-Uniform"
                    elif grid_name == 'SPARSE':
                        method_name = 'Smolyak'
                    else:
                        raise ValueError(f"Can't handle grid: {grid_name}")

                    if dim_index == 0:
                        if grid_index == 0:
                            output[scale_name] += r"\multirow{3}{*}{\thead[l]{\tiny\textbf{" + abbreviation_dict[
                                str(fun_name)] + r"}\\" + r"$Q=" + str(min(n_functions_list)) + r"$}} & "
                        else:
                            output[scale_name] += r" & "
                        output[scale_name] += method_name
                    else:
                        output[scale_name] += r" "

                    ell_2_error = dim_df['ell_2_error']
                    ell_infty_error = dim_df['ell_infty_error']
                    ell_2_max = ell_2_error.max()
                    ell_infty_max = ell_infty_error.max()

                    if not skip_mean_error:
                        ell_2_mean = ell_2_error.mean()
                        ell_infty_mean = ell_infty_error.mean()

                    if skip_mean_error:
                        output[scale_name] += (
                                r" & "
                                + highlight_matching_value(ell_2_max, min_values[str(dim)]['ell_2_max'])
                                + r" & "
                                + highlight_matching_value(ell_infty_max, min_values[str(dim)]['ell_infty_max'])
                        )
                    else:
                        output[scale_name] += (
                                r" & "
                                + highlight_matching_value(ell_2_mean, min_values[str(dim)]['ell_2_mean'])
                                + r" & "
                                + highlight_matching_value(ell_infty_mean, min_values[str(dim)]['ell_infty_mean'])
                                + r" & "
                                + highlight_matching_value(ell_2_max, min_values[str(dim)]['ell_2_max'])
                                + r" & "
                                + highlight_matching_value(ell_infty_max, min_values[str(dim)]['ell_infty_max'])
                        )

                output[scale_name] += r"\\" + "\n"
            if not fun_index == len(dfg) - 1:
                output[scale_name] += r"\midrule" + "\n"
        output[scale_name] += r"\bottomrule" + "\n"
        output[scale_name] += r"\end{tabular}" + "\n"

    for scale_name, table in output.items():
        with open(os.path.join(output_folder, f"scale{scale_name}.tex"), "w") as f:
            f.write(table)
        print("Exported table for scale", scale_name, "to", os.path.join(output_folder, f"scale{scale_name}.tex"))


def generate_table_fixed_fun(results_csv_path: str, output_folder: str, skip_mean_error: bool = False,
                             skip_scale: Union[List, None] = None):
    r""" Creates a tex file for each dimension in the specified csv file. The tex files will be stored in the given folder.

        In the LaTeX File, the table can be printed via
        \begin{table}[htbp]
            \label{tab:dim4_results}
            \centering
            \begin{adjustbox}{width=\linewidth}
                \input{>>TEX-FILE-PATH<<}
            \end{adjustbox}
            \vspace{0.1cm}
            \caption{Test\label{tab:dim4_results}}
        \end{table}

        For that all the following packages are needed
        % Packages for the tabulars
        \usepackage{booktabs}
        \usepackage{array}
        \usepackage{adjustbox}
        \usepackage{multirow}
        \usepackage{makecell}

        Additionally, we need the following commands
        % Command for the first entry in a row
        \newcommand{\first}[1]{\textbf{#1}}
    """

    output = dict()

    error_reductions = ['max']

    if not skip_mean_error:
        error_reductions.insert(0, 'mean')
        errors = [r'e_{\rm mean}', r'e_{\rm max}']
    else:
        errors = [r'e_{\rm mean}^{\rm wc}', r'e_{\rm max}^{\rm wc}']

    no_error_combinations = len(errors) * len(error_reductions)

    results = pd.read_csv(results_csv_path, sep=',', header=0, decimal='.')

    for fun_name, fun_df in results.groupby('f_name'):

        if skip_scale is None:
            skip_scale = list()

        # get scales
        scales = sorted(fun_df['scale'].unique())
        scales = [s for s in scales if s not in skip_scale]

        right_text = ("|" + ("r" * no_error_combinations)) * len(scales)

        output[fun_name] = f"% Created with Python on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "\n"
        output[
            fun_name] += f"% {results_csv_path}, function = {fun_name}, scales = {[int(s) for s in scales]}" + "\n"
        output[fun_name] += r"\begin{tabular}{ll" + right_text + r"|}" + "\n"

        # add header
        output[fun_name] += r" &  "
        for i, scale in enumerate(scales):
            if i == 0:
                output[fun_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                    no_error_combinations) + r"}{c}{Scale" + str(scale) + r"}"
            else:
                output[fun_name] += r" & \multicolumn{" + str(no_error_combinations) + r"}{c}{Scale" + str(scale) + r"}"
        output[fun_name] += r"\\" + "\n"

        if len(error_reductions) > 1:
            output[fun_name] += r" &  "
            for i, scale in enumerate(scales):
                for j, error_reduction in enumerate(error_reductions):
                    if i == 0 and j == 0:
                        output[fun_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                            len(errors)) + r"}{c}{" + str(
                            error_reduction) + r"}"
                    else:
                        output[fun_name] += r" & \multicolumn{" + str(len(errors)) + r"}{c}{" + str(
                            error_reduction) + r"}"
            output[fun_name] += r"\\" + "\n"

        output[fun_name] += r" &  "
        for i, scale in enumerate(scales):
            for j, error_reduction in enumerate(error_reductions):
                for k, error in enumerate(errors):
                    if i == 0 and j == 0 and k == 0:
                        output[fun_name] += r" \multicolumn{1}{c}{} & \multicolumn{1}{c}{$" + error + r"$}"
                    else:
                        output[fun_name] += r" & \multicolumn{1}{c}{$" + error + r"$}"
        output[fun_name] += r"\\" + "\n" + r"\toprule" + "\n"

        dfg = fun_df.groupby('dim')
        for dim_index, (dim_name, dim_df) in enumerate(dfg):
            min_values = {}

            n_functions_list = []

            for i, scale in enumerate(scales):
                scale_df = dim_df[dim_df['scale'] == scale]

                if len(scale_df) == 0:  # No experiment for this scale
                    ell_2_max = [42]  # dummy value
                    ell_infty_max = [42]  # dummy value
                    ell_2_mean = [42]  # dummy value
                    ell_infty_mean = [42]  # dummy value

                else:
                    ell_2_max = []
                    ell_infty_max = []

                    if not skip_mean_error:
                        ell_2_mean = []
                        ell_infty_mean = []

                    for method_name, method_df in scale_df.groupby('grid_type', sort=False):

                        n_functions_list.append(len(method_df))

                        ell_2_max.append(method_df['ell_2_error'].max())
                        ell_infty_max.append(method_df['ell_infty_error'].max())

                        if not skip_mean_error:
                            ell_2_mean.append(method_df['ell_2_error'].mean())
                            ell_infty_mean.append(method_df['ell_infty_error'].mean())

                min_values[str(scale)] = {
                    'ell_2_max': min(ell_2_max),
                    'ell_infty_max': min(ell_infty_max),
                }

                if not skip_mean_error:
                    min_values[str(scale)] = {
                        'ell_2_max': min(ell_2_max),
                        'ell_infty_max': min(ell_infty_max),
                        'ell_2_mean': min(ell_2_mean),
                        'ell_infty_mean': min(ell_infty_mean),
                    }

            for grid_index, (grid_name, grid_df) in enumerate(dim_df.groupby('grid_type', sort=False)):

                for scale_index, scale_name in enumerate(scales):
                    scale_df = grid_df[grid_df['scale'] == scale_name]

                    if grid_name == 'CHEBYSHEV':
                        method_name = "LS-Chebyshev"
                    elif grid_name == 'UNIFORM':
                        method_name = "LS-Uniform"
                    elif grid_name == 'SPARSE':
                        method_name = 'Smolyak'
                    else:
                        raise ValueError(f"Can't handle grid: {grid_name}")

                    if scale_index == 0:
                        if grid_index == 0:
                            output[fun_name] += r"\multirow{3}{*}{\thead[l]{\textbf{Dim " + str(
                                dim_name) + r"}\\" + r"$Q=" + str(min(n_functions_list)) + r"$}} & "
                        else:
                            output[fun_name] += r" & "
                        output[fun_name] += method_name
                    else:
                        output[fun_name] += r" "

                    if len(scale_df) == 0:
                        if skip_mean_error:
                            output[fun_name] += (r" & " + "" + r" & " + "")
                        else:
                            output[fun_name] += (
                                    r" & " + "" + r" & " + "" + r" & " + "" + r" & " + "")
                    else:
                        ell_2_error = scale_df['ell_2_error']
                        ell_infty_error = scale_df['ell_infty_error']

                        ell_infty_max = ell_infty_error.max()
                        ell_2_max = ell_2_error.max()

                        if not skip_mean_error:
                            ell_infty_mean = ell_infty_error.mean()
                            ell_2_mean = ell_2_error.mean()

                        if skip_mean_error:
                            output[fun_name] += (
                                    r" & "
                                    + highlight_matching_value(ell_2_max, min_values[str(scale_name)]['ell_2_max'])
                                    + r" & "
                                    + highlight_matching_value(ell_infty_max,
                                                               min_values[str(scale_name)]['ell_infty_max'])
                            )
                        else:
                            output[fun_name] += (
                                    r" & "
                                    + highlight_matching_value(ell_2_mean, min_values[str(scale_name)]['ell_2_mean'])
                                    + r" & "
                                    + highlight_matching_value(ell_infty_mean,
                                                               min_values[str(scale_name)]['ell_infty_mean'])
                                    + r" & "
                                    + highlight_matching_value(ell_2_max, min_values[str(scale_name)]['ell_2_max'])
                                    + r" & "
                                    + highlight_matching_value(ell_infty_max,
                                                               min_values[str(scale_name)]['ell_infty_max'])
                            )

                output[fun_name] += r"\\" + "\n"
            if not dim_index == len(dfg) - 1:
                output[fun_name] += r"\midrule" + "\n"
        output[fun_name] += r"\bottomrule" + "\n"
        output[fun_name] += r"\end{tabular}" + "\n"

        # remove last occurence of midrule

    for fun_name, table in output.items():
        with open(os.path.join(output_folder, f"{fun_name}.tex"), "w") as f:
            f.write(table)
        print("Exported table for ", fun_name, "to", os.path.join(output_folder, f"{fun_name}.tex"))


if __name__ == '__main__':
    input_path = "path/to/your/results_numerical_experiments.csv"
    output_folder = os.path.join("..", "results", "final_results", "tables")
    os.makedirs(output_folder, exist_ok=True)

    ignore_scale = {
        "2": [1, 2],
        "3": [1, 2],
        "4": [1, 2],
        "5": [1, 2],
        "6": [1, 2],
        "7": [1, 2],
        "8": [1, 2],
        "9": [1, 2],
        "10": [1, 2],
    }

    # ignore_dim = None
    # ignore_scale = None

    generate_table_fixed_dim(input_path, output_folder, skip_mean_error=True, skip_scale=ignore_scale)
    generate_table_fixed_scale(input_path, output_folder, skip_mean_error=True, skip_dim=None)
    generate_table_fixed_fun(input_path, output_folder, skip_mean_error=True, skip_scale=[1, 2])
