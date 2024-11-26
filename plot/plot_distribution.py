from typing import Union, List

import pandas as pd
import os

from function import FunctionType
import matplotlib.pyplot as plt
import numpy as np


# def plot_all_errors(dimension, seed: int, function_type: FunctionType, scales: List[int], multiplier_fun: Callable,
#                     folder_name: str, path: Union[str, None] = None, save: bool = False,
#                     save_path: Union[str, None] = None, same_axis_both_plots: bool = True):

def plot_all_errors(file_name: str):
    df = pd.read_csv(file_name, header=0, sep=',', decimal='.')

    # get distinct values for dimension
    dimensions = df['dim'].unique()

    # get distinct values for function type
    function_types = df['f_name'].unique()

    for f_type in function_types:
        for dim in dimensions:
            # select the data
            data = df[(df['dim'] == dim) & (df['f_name'] == f_type)].copy()
            data.drop(['datetime', 'needed_time', 'sum_c', 'f_name'], axis=1, inplace=True)
            print(data)

    labels = ['C_r 03', 'C_r 05', 'C_r 0.1', 'C_r 0.2', 'C_r 0.5', 'C_r 1', 'C_r 2', 'Unconfined']

    my_list = [np.random.uniform(10, 30, 5) for _ in labels]
    my_mean = [values.mean() for values in my_list]

    plt.plot(np.arange(len(my_mean)) + 1, my_mean, color='r')
    plt.boxplot(my_list, labels=labels)
    plt.show()


if __name__ == '__main__':

    file_name = os.path.join("results", "26_11_2024_13_05_23", "results_numerical_experiments.csv")

    plot_all_errors(file_name)


