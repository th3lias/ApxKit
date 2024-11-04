import numpy as np
import pandas as pd

from fit import SmolyakFitter
from function import SmolyakModel, Function, FunctionType, ParametrizedFunctionProvider
from grid import RuleGridProvider


class SmolyakExperimentExecutor:
	def __init__(self, dim_list: list[int], scale_list: list[int], num_test_points: int,
	             path: str = "results/current_results.csv"):
		"""
			An object that can execute a series of experiments on Smolyak models.
		"""
		self.dim_list = dim_list
		self.scale_list = scale_list
		self.num_test_points = num_test_points
		# Ensure that any used directory is created before this object is created, otherwise this fails.
		self.results_path = path
		header = {
			"function": [],
			"dim": [],
			"scale": [],
			"l_2": [],
			"l_inf": []
		}
		df = pd.DataFrame(header)
		df.to_csv(self.results_path, index=False)
	
	def execute_random_experiments(self, function_types: list[FunctionType]):
		for dim in self.dim_list:
			fitter = SmolyakFitter(dim)
			rule_grid_provider = RuleGridProvider(dim)
			test_points = np.random.uniform(0, 1, (self.num_test_points, dim))
			self.execute_single_dim_experiment(dim, function_types, fitter, rule_grid_provider, test_points)
	
	def execute_single_dim_experiment(self, dim: int, function_types: list[FunctionType], fitter: SmolyakFitter,
	                                  rule_grid_provider: RuleGridProvider, test_points: np.ndarray):
		c = np.random.uniform(0, 1, dim)
		c = c / np.sum(c) * dim
		w = np.random.uniform(0, 1, dim)
		for function_type in function_types:
			function = ParametrizedFunctionProvider.get_function(function_type, dim, c=c, w=w)
			self.execute_single_function_experiment(function, fitter, rule_grid_provider, test_points)
	
	def execute_single_function_experiment(self, function: Function, fitter: SmolyakFitter,
	                                       rule_grid_provider: RuleGridProvider, test_points):
		for scale in self.scale_list:
			grid = rule_grid_provider.generate(scale)
			model = fitter.fit(function, grid)
			l_2, l_inf = self.evaluate_model(function, model, test_points)
			self.save_stats(function, scale, l_2, l_inf)
	
	@staticmethod
	def evaluate_model(function: Function, model: SmolyakModel, test_points: np.ndarray):
		"""
			Evaluate the model on the test points and return the L2 and L_inf norms.
		"""
		model_values = model(test_points).reshape(-1, 1)
		true_values = function(test_points).reshape(-1, 1)
		l_2 = np.sum(np.square(model_values - true_values)) / function.dim
		l_inf = np.max(np.abs(model_values - true_values))
		return l_2, l_inf
	
	def save_stats(self, function: Function, scale: int, l_2: float, l_inf: float):
		"""
			Keep the CSV up to date with the current results.
		"""
		data = {
			"function": [function.name],
			"dim": [function.dim],
			"scale": [scale],
			"l_2": [l_2],
			"l_inf": [l_inf]
		}
		df = pd.DataFrame(data)
		df.to_csv(self.results_path, mode='a', header=False, index=False)


if __name__ == "__main__":  # Example usage
	dims = [2, 3, 4, 5, 6]
	scales = [2, 3, 4, 5]
	num_points = 10000
	experiment_executor = SmolyakExperimentExecutor(dims, scales, num_points)
	functions = [FunctionType.OSCILLATORY, FunctionType.GAUSSIAN, FunctionType.G_FUNCTION]
	experiment_executor.execute_random_experiments(functions)
