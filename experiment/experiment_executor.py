import datetime
import os.path
import platform
import time

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from algorithm.algorithm import Algorithm
from function import FunctionType, ParametrizedFunctionProvider
from grid.generator.grid_generator import GridGenerator
from grid.generator.uniform_number_generator import UniformNumberGenerator
import torch


class ExperimentExecutor:
    """Orchestrates the full benchmark: fits algorithms on test functions across dimension/scale combinations."""

    def __init__(self, dim_scale_dict: dict[int, list[int]], test_grid_generator: GridGenerator,
                 uniform_value_generator_c: UniformNumberGenerator, uniform_value_generator_w: UniformNumberGenerator,
                 use_max_scale: bool, path: str = None):
        current_datetime = datetime.datetime.now()

        if path is None:
            self.results_path = os.path.join("results", current_datetime.strftime('%d_%m_%Y_%H_%M_%S'),
                                             "results_numerical_experiments.csv")
        else:
            self.results_path = path

        # Ensure scales are sorted and deduplicated per dimension
        for dim in dim_scale_dict.keys():
            dim_scale_dict[dim] = sorted(list(set(dim_scale_dict[dim])))
        self.dim_scale_dictionary = dim_scale_dict
        self.test_grid_generator = test_grid_generator
        self.uniform_value_generator_c = uniform_value_generator_c
        self.uniform_value_generator_w = uniform_value_generator_w
        self.use_max_scale = use_max_scale

        self.header_keys = ['dim', 'scale', 'algorithm', 'abbr_algorithm', 'method', 'abbr_method', 'basis_name',
                            'abbr_basis_name', 'grid_name', 'abbr_grid_name', 'w', 'c', 'sum_c',
                            'n_samples', 'n_test_samples', 'seed_list', 'f_name', 'ell_2_error', 'ell_infty_error',
                            'datetime', 'needed_time']

        self.functions = None
        self.test_functions = None
        self.cs = None
        self.ws = None
        self.f_names = None

        self.results_df = pd.DataFrame(columns=self.header_keys)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute_experiments(self, algorithms: list[Algorithm] | Algorithm,
                            function_types: list[FunctionType] | FunctionType, n_functions_parallel: int,
                            avg_c: dict | float, seed_list: list[int], device: torch.device,
                            plot_callbacks: list = None) -> pd.DataFrame:
        """
        Run all (dimension, scale, algorithm) combinations and collect errors.

        Parameters
        ----------
        plot_callbacks : list[callable], optional
            Each callback receives ``(results_df, save_dir)`` after the full run.

        Returns
        -------
        pd.DataFrame   Accumulated results.
        """
        if isinstance(algorithms, Algorithm):
            algorithms = [algorithms]
        if plot_callbacks is None:
            plot_callbacks = []

        save_dir = os.path.dirname(self.results_path)

        print(
            f"Starting experiment with:\n" +
            "_" * 75 + "\n" +
            f"* algorithms:{[algo.name for algo in algorithms]}\n" +
            f"* dimension/scale {self.dim_scale_dictionary}\n" +
            f"* n_functions={n_functions_parallel}\n" +
            f"* seed_list={seed_list}\n" +
            f"* cpu={platform.processor()}\n" +
            f"* RAM={psutil.virtual_memory().total / 1024 / 1024 / 1024} GB\n" +
            f"* random test rule: {self.test_grid_generator.name}\n" +
            f"* max_scale={self.use_max_scale}\n" +
            f"* device if needed: {device})\n" +
            f"* starting time: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} ")
        print("_" * 75)
        print(f"Results will be stored in {self.results_path}\n")

        time.sleep(1)

        n_iterations = sum(len(scales) for scales in self.dim_scale_dictionary.values())
        total_iterations = n_iterations * len(algorithms)
        progress_bar = tqdm(total=total_iterations, desc="Initializing", unit="iteration")

        for dim in self.dim_scale_dictionary.keys():
            self._get_functions(function_types, n_functions_parallel, dim, avg_c)
            max_scale = max(self.dim_scale_dictionary[dim])

            # Pre-evaluate test function values (reused across scales if use_max_scale)
            if self.use_max_scale:
                test_grid = self.test_grid_generator.get_grid(dim=dim, scale=max_scale,
                                                              lower_bound=0.0, upper_bound=1.0)
                n_points_test = test_grid.get_num_points()
                y_test = np.empty(dtype=np.float64, shape=(n_points_test, len(self.test_functions)))
                for i, test_function in enumerate(self.test_functions):
                    y_test[:, i] = test_function(test_grid.grid)

            for scale in self.dim_scale_dictionary.get(dim):
                if not self.use_max_scale:
                    test_grid = self.test_grid_generator.get_grid(dim=dim, scale=scale,
                                                                  lower_bound=0.0, upper_bound=1.0)
                    n_points_test = test_grid.get_num_points()
                    y_test = np.empty(dtype=np.float64, shape=(n_points_test, len(self.test_functions)))
                    for i, test_function in enumerate(self.test_functions):
                        y_test[:, i] = test_function(test_grid.grid)

                for algo in algorithms:
                    abbr = f"{algo.abbr_name}_{algo.solver.abbr_name}_{algo.basis_generator.abbr_name}_{algo.grid_generator.abbr_name}"
                    progress_bar.set_description(f"d={dim} s={scale} {abbr}")

                    start_time = time.time()
                    algo.fit(dim=dim, scale=scale, f=self.functions, lower=0.0, upper=1.0)
                    y_hat_test = algo.evaluate(test_grid)
                    ell_2, ell_infty = self._calc_error(y_test, y_hat_test)
                    needed_time = time.time() - start_time

                    self._save_stats(
                        dim=dim, scale=scale,
                        algorithm=algo.name, abbr_algorithm=algo.abbr_name,
                        method=algo.solver.name, abbr_method=algo.solver.abbr_name,
                        grid_name=algo.grid_generator.name, abbr_grid_name=algo.grid_generator.abbr_name,
                        basis_name=algo.basis_generator.name, abbr_basis_name=algo.basis_generator.abbr_name,
                        seed_list=seed_list,
                        ell_2_errors=ell_2, n_points_fit=algo.get_n_points(), n_points_test=n_points_test,
                        ell_infty_errors=ell_infty,
                        date_time=datetime.datetime.now(),
                        needed_time=round(needed_time, 3))

                    self._flush_csv()
                    progress_bar.update(1)

                # Plot callbacks
                for cb in plot_callbacks:
                    cb(self.results_df, save_dir, d=dim, s=scale)

        progress_bar.close()

        for cb in plot_callbacks:
            cb(self.results_df, save_dir, d=None, s=None)

        print(f"Done at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        return self.results_df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_csv(self):
        """Write the accumulated results to CSV."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        self.results_df.to_csv(self.results_path, index=False, sep=',', decimal='.', header=True)

    def _get_functions(self, function_types: list[FunctionType] | FunctionType, n_functions_parallel: int, dim: int,
                       avg_c: dict | float):
        """Build lists of training and test functions for the given dimension."""
        if isinstance(function_types, FunctionType):
            function_types = [function_types]

        functions, test_functions = [], []
        cs, ws, f_names = [], [], []

        for fun_type in function_types:
            avg_c_fun = avg_c[fun_type] if isinstance(avg_c, dict) else avg_c
            if isinstance(avg_c, dict) and fun_type not in avg_c:
                raise ValueError(f"Function type {fun_type} not in average c dictionary")

            for i in range(n_functions_parallel):
                c, w = self._get_c_and_w(avg_c_fun, dim)
                functions.append(ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w))
                test_functions.append(ParametrizedFunctionProvider.get_function(fun_type, dim, c=c, w=w, test=True))
                cs.append(c)
                ws.append(w)
                f_names.append(fun_type.name)

        self.functions = functions
        self.test_functions = test_functions
        self.cs = cs
        self.ws = ws
        self.f_names = f_names

    def _get_c_and_w(self, avg_c: float, dim: int):
        """Draw random c and w, normalise c to have mean avg_c, then reshuffle generators."""
        w = self.uniform_value_generator_w.get_random_numbers(lower_bound=0.0, upper_bound=1.0, n_points=dim, dim=1)
        c = self.uniform_value_generator_c.get_random_numbers(lower_bound=0.0, upper_bound=1.0, n_points=dim, dim=1)

        self.uniform_value_generator_w.reshuffle()
        self.uniform_value_generator_c.reshuffle()

        c = c / np.sum(c) * dim * avg_c
        return np.atleast_1d(c), np.atleast_1d(w)

    def _save_stats(self, dim: int, scale: int, algorithm: str, abbr_algorithm: str, method: str, abbr_method: str,
                    grid_name: str, abbr_grid_name: str, basis_name: str, abbr_basis_name: str, seed_list: list[int],
                    n_points_fit: int, n_points_test: int, ell_2_errors: np.ndarray | list[float],
                    ell_infty_errors: np.ndarray | list[float], date_time: datetime.datetime, needed_time: float):
        """Append one row per function to the in-memory results DataFrame."""
        if isinstance(ell_2_errors, np.ndarray):
            ell_2_errors = ell_2_errors.tolist()
        if isinstance(ell_infty_errors, np.ndarray):
            ell_infty_errors = ell_infty_errors.tolist()

        n_functions = len(ell_2_errors)
        seed_list_str = "_".join(str(s) for s in seed_list)

        formatted_cs = [np.array2string(c, precision=5, separator=',', suppress_small=True).replace('\n', '')
                        for c in self.cs]
        formatted_ws = [np.array2string(w, precision=5, separator=',', suppress_small=True).replace('\n', '')
                        for w in self.ws]

        data = {
            'dim': [dim] * n_functions,
            'scale': [scale] * n_functions,
            'algorithm': [algorithm] * n_functions,
            'abbr_algorithm': [abbr_algorithm] * n_functions,
            'method': [method] * n_functions,
            'abbr_method': [abbr_method] * n_functions,
            'basis_name': [basis_name] * n_functions,
            'abbr_basis_name': [abbr_basis_name] * n_functions,
            'grid_name': [grid_name] * n_functions,
            'abbr_grid_name': [abbr_grid_name] * n_functions,
            'w': formatted_ws,
            'c': formatted_cs,
            'sum_c': [round(np.sum(c), 3) for c in self.cs],
            'n_samples': [n_points_fit] * n_functions,
            'n_test_samples': [n_points_test] * n_functions,
            'seed_list': [seed_list_str] * n_functions,
            'f_name': self.f_names,
            'ell_2_error': ell_2_errors,
            'ell_infty_error': ell_infty_errors,
            'datetime': [date_time] * n_functions,
            'needed_time': [needed_time] * n_functions,
        }

        self.results_df = pd.concat([self.results_df, pd.DataFrame(data)], ignore_index=True)

    @staticmethod
    def _calc_error(true_array, approximated_array):
        """Compute per-function ℓ₂ and ℓ∞ errors (axis=0 averages over points)."""
        ell_2 = np.sqrt(np.mean(np.square(true_array - approximated_array), axis=0))
        ell_infty = np.max(np.abs(true_array - approximated_array), axis=0)

        return ell_2, ell_infty
