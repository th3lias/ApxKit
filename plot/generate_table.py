import datetime

import pandas as pd
import numpy as np
import os


# TODO: Also add the needed time and the number of points to the table

def generate_table(results_csv_path: str, output_folder: str):
    output = dict()

    errors = [r'\ell_2', r'\ell_\infty']
    error_reductions = ['mean',
                        'max']  # do not swap unless you are sure that the results in the csv are in the correct order

    no_error_combinations = len(errors) * len(error_reductions)

    results = pd.read_csv(results_csv_path, sep=',', header=0, decimal='.')

    for dim_name, dim_df in results.groupby('dim'):

        # get max scale
        max_scale = dim_df['scale'].max()
        min_scale = dim_df['scale'].min()

        assert min_scale == 1, "Scale needs to start at 1"

        right_text = ("|" + ("r" * no_error_combinations)) * max_scale

        output[dim_name] = f"% Created with Python on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "\n"

        output[dim_name] += f"% {results_csv_path}, dim={dim_name}" + "\n"

        output[dim_name] += r"\begin{tabular}{ll" + right_text + r"|}" + "\n"

        # add header
        output[dim_name] += r" &  "
        for i in range(1, max_scale + 1):
            if i == 1:
                output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(
                    no_error_combinations) + r"}{c}{Scale" + str(i) + r"}"
            else:
                output[dim_name] += r" & \multicolumn{" + str(no_error_combinations) + r"}{|c}{Scale" + str(i) + r"}"
        output[dim_name] += r"\\" + "\n"

        output[dim_name] += r" &  "
        for i in range(1, max_scale + 1):
            for j, error_reduction in enumerate(error_reductions):
                if i == 1 and j == 0:
                    output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{" + str(len(errors)) + r"}{c}{" + str(
                        error_reduction) + r"}"
                else:
                    output[dim_name] += r" & \multicolumn{" + str(len(errors)) + r"}{c}{" + str(error_reduction) + r"}"
        output[dim_name] += r"\\" + "\n"

        output[dim_name] += r" &  "
        for i in range(1, max_scale + 1):
            for j, error_reduction in enumerate(error_reductions):
                for k, error in enumerate(errors):
                    if i == 1 and j == 0 and k == 0:
                        output[dim_name] += r" \multicolumn{1}{c}{} & \multicolumn{1}{c}{$" + error + r"$}"
                    else:
                        output[dim_name] += r" & \multicolumn{1}{c}{$" + error + r"$}"
        output[dim_name] += r"\\\toprule\\" + "\n"

        for function_index, (fun_name, fun_df) in enumerate(dim_df.groupby('f_name')):

            for method_index, (method_name, method_df) in enumerate(fun_df.groupby(['method', 'grid_type'])):

                for scale_index, (scale_name, scale_df) in enumerate(method_df.groupby('scale')):

                    if method_name == ('Least_Squares', 'CHEBYSHEV'):
                        str_method_name = "LS-Chebyshev"
                    elif method_name == ('Least_Squares', 'UNIFORM'):
                        str_method_name = "LS-Uniform"
                    elif method_name == ('Smolyak', 'CHEBYSHEV'):
                        str_method_name = 'Smolyak'
                    else:
                        raise ValueError(f"Can't handle Method: {method_name}")

                    if scale_index == 0:
                        if method_index == 0:
                            output[dim_name] += r"\multirow{3}{*}{{\rotatebox[origin=c]{90}{\textbf{" + str(
                                fun_name).replace("_", "-") + r"}}}} & "
                        else:
                            output[dim_name] += r" & "
                        output[dim_name] += str_method_name
                    else:
                        output[dim_name] += r" "

                    # get max_ell_infty error, mean_ell_infty error, max_ell_2 error, mean_ell_2 error
                    ell_2_error = scale_df['ell_2_error']
                    ell_infty_error = scale_df['ell_infty_error']

                    ell_infty_mean = ell_infty_error.mean()
                    ell_infty_max = ell_infty_error.max()
                    ell_2_mean = ell_2_error.mean()
                    ell_2_max = ell_2_error.max()

                    output[
                        dim_name] += r" & " + f"{ell_2_mean:.2e} & {ell_infty_mean:.2e} & {ell_2_max:.2e} & {ell_infty_max:.2e}"
                output[dim_name] += r"\\" + "\n"
            # if not last function:
            if fun_name != dim_df['f_name'].iloc[-1]:
                output[dim_name] += r"\midrule" + "\n"
            else:
                output[dim_name] += r"\bottomrule" + "\n"
        output[dim_name] += r"\end{tabular}" + "\n"

    for dim_name, table in output.items():
        with open(os.path.join(output_folder, f"dim{dim_name}.tex"), "w") as f:
            f.write(table)
    print("Done")


if __name__ == '__main__':
    input_path = os.path.join("..", "results", "23_11_2024_14_36_53", "results_numerical_experiments.csv")
    output_folder = os.path.join("..", "paper", "tables")

    generate_table(input_path, output_folder)
