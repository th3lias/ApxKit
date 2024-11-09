import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.experiment_executor import ExperimentExecutor
from fit import SmolyakFitter
from function import SmolyakModel, Function, FunctionType, ParametrizedFunctionProvider
from function.parametrized_f import ParametrizedFunction
from grid import RuleGridProvider


# TODO: Adapt that to the LS model
# TODO: Main Method needs also to adapt to this behaviour
# TODO: Make a parent class for the experiment executor
# TODO: Add tqdm progressbar
# TODO: Delete this (and LeastSquaresExperimentExecutor) in the end
class SmolyakExperimentExecutor(ExperimentExecutor):
    def __init__(self, dim_list: list[int], scale_list: list[int], num_test_points: int, path: str):
        """
            An object that can execute a series of experiments on Smolyak models.
        """
        super().__init__(dim_list, scale_list, num_test_points, path)

    def execute_random_experiments(self, function_types: list[FunctionType]):
        for dim in self.dim_list:
            fitter = SmolyakFitter(dim)
            rule_grid_provider = RuleGridProvider(dim)
            test_points = np.random.uniform(0, 1, size=(self.num_test_points, dim)).astype(np.float64)
            self.execute_single_dim_experiment(dim, function_types, fitter, rule_grid_provider, test_points)
        self.plot_csv()

    # TODO: Check whether we can make it faster by not fitting the function n times
    def execute_single_dim_experiment(self, dim: int, function_types: list[FunctionType], fitter: SmolyakFitter,
                                      rule_grid_provider: RuleGridProvider, test_points: np.ndarray):
        c = np.random.uniform(0, 1, dim).astype(np.float64)
        c = c / np.sum(c) * dim
        w = np.random.uniform(0, 1, dim).astype(np.float64)
        for function_type in function_types:
            function = ParametrizedFunctionProvider.get_function(function_type, dim, c=c, w=w)
            self.execute_single_function_experiment(function, fitter, rule_grid_provider, test_points)

    def execute_single_function_experiment(self, function: ParametrizedFunction, fitter: SmolyakFitter,
                                           rule_grid_provider: RuleGridProvider, test_points):
        for scale in self.scale_list:
            grid = rule_grid_provider.generate(scale)
            model = fitter.fit(function, grid)
            l_2, l_inf = self.evaluate_model(function, model, test_points)
            self.save_stats(function, scale, l_2, l_inf)

    def exectute_single_

    @staticmethod
    def evaluate_model(function: Function, model: SmolyakModel, test_points: np.ndarray) -> tuple[float, float]:
        """
            Evaluate the model on the test points and return the L2 and L_inf norms.
        """
        model_values = model(test_points).reshape(-1, 1).astype(np.float64)
        true_values = function(test_points).reshape(-1, 1).astype(np.float64)
        l_2 = np.sqrt(np.mean(np.square(model_values - true_values)))
        l_inf = np.max(np.abs(model_values - true_values))
        return l_2, l_inf

    def save_stats(self, function: ParametrizedFunction, scale: int, l_2: float, l_inf: float):
        """
            Keep the CSV up to date with the current results.
        """
        data = {
            "function": [function.name],
            "dim": [function.dim],
            "scale": [scale],
            "c": str(function.c),
            "w": str(function.w),
            "l_2": [l_2],
            "l_inf": [l_inf]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.results_path, mode='a', header=False, index=False)

    def plot_csv(self):
        """
            Plot the results from the CSV file using plotly.
            Plot Scale vs Error.
        """
        plt.figure(figsize=(10, 5))
        df = pd.read_csv(self.results_path)
        for function in df['function'].unique():
            df_function = df[df['function'] == function]
            plt.plot(df_function['scale'], df_function['l_2'], label=function)
        plt.xlabel("Scale")
        plt.ylabel("L2 Norm")
        plt.legend()
        plt.show()

# TODO: Remove this as soon as we are finished with the refactoring
if __name__ == "__main__":  # Example usage
    dims = [3, 4, 5]
    scales = [4, 5, 6]
    num_points = 1000
    experiment_executor = SmolyakExperimentExecutor(dims, scales, num_points, os.path.join("..", "results", "temp.csv"))
    functions = [
        FunctionType.OSCILLATORY,
        FunctionType.PRODUCT_PEAK,
        FunctionType.GAUSSIAN,
        # FunctionType.CORNER_PEAK,  --- Tasmanian has huge problems with this function
        FunctionType.CONTINUOUS,
        FunctionType.DISCONTINUOUS,
        FunctionType.G_FUNCTION,
        # FunctionType.MOROKOFF_CALFISCH_1,  --- The calculation of this function always throws a warning
        FunctionType.MOROKOFF_CALFISCH_2,
        # FunctionType.ROOS_ARNOLD,  --- Tasmanian has quite some problems with this function
        FunctionType.BRATLEY,
        FunctionType.ZHOU
    ]
    experiment_executor.execute_random_experiments(functions)
