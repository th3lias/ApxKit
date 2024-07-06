import os
from typing import Callable, Union, List, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import lsmr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from grid.grid import Grid
from grid.grid_type import GridType
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator
from interpolate.interpolation_methods import LeastSquaresMethod
from nn.dataset_torch import LSDataset
from nn.nn_torch import LSNN
from nn.nn_train import train
from utils.utils import find_degree
import padasip as pa

from scipy.linalg import lu


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, include_bias: bool, basis_type: BasisType, method: LeastSquaresMethod = LeastSquaresMethod.EXACT,
                 grid: Union[Grid, None] = None):
        super().__init__(grid)
        self.include_bias = include_bias
        self.method = method
        self.basis_type = basis_type

    def set_method(self, method: LeastSquaresMethod):
        self.method = method

    def fit(self, f: Union[Callable, List[Callable]]):

        assert self.grid is not None, "Grid needs to be set before interpolation"

        if self.basis is None:
            self.basis = self._build_basis()

        if self.method == LeastSquaresMethod.ITERATIVE_LSMR:
            self._approximate_lsmr(f)
        elif self.method == LeastSquaresMethod.EXACT:
            self._approximate_exact(f)
        elif self.method == LeastSquaresMethod.SKLEARN:
            self._approximate_sklearn(f)
        elif self.method == LeastSquaresMethod.PYTORCH:
            self._approximate_pt(f)
        elif self.method == LeastSquaresMethod.PYTORCH_NEURAL_NET:
            self._approximate_nn(f)
        elif self.method == LeastSquaresMethod.JAX_NEURAL_NET:
            raise NotImplementedError
        elif self.method == LeastSquaresMethod.RLS:
            self._approximate_rls(f)
        elif self.method == LeastSquaresMethod.ITERATIVE_RLS:
            self._approximate_iterative_rls(f)
        elif self.method == LeastSquaresMethod.NUMPY_LSTSQ:
            self._approximate_numpy_lstsq(f)
        else:
            raise ValueError(f'The method {self.method.name} is not supported')

    def interpolate(self, grid: Union[Grid, np.ndarray]):
        """
        Applies the fitted function on the given data and returns the calculates values
        :param grid: Grid on which the fitted function should be tested
        """

        if isinstance(grid, Grid):
            grid = grid.grid

        data_pol = self._build_basis(basis_type=None, grid=grid, b_idx=self._b_idx)
        y_hat = data_pol @ self.coeff
        if y_hat.ndim > 1:
            return y_hat.T
        return y_hat

    def _build_basis(self, basis_type: Union[BasisType, None] = None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV and not basis_type == BasisType.REGULAR:
            raise ValueError(f"Unsupported Basis-Type. Expected Chebyshev or Regular Basis but got {basis_type}")

        if grid is None:
            grid = self.grid.grid

        if basis_type == BasisType.CHEBYSHEV:
            return self._build_poly_basis(grid, b_idx)

        elif basis_type == BasisType.REGULAR:
            print(DeprecationWarning("This will not be supported in the near future"))
            degree = find_degree(self.scale, self.dim)

            poly = PolynomialFeatures(degree=degree, include_bias=self.include_bias)
            return poly.fit_transform(grid)

    def _approximate_exact(self, f: Union[Callable, List[Callable]]):
        """
        Approximates a (or multiple) function(s) with polynomials by least squares.
        :param f: function or list of functions that need to be approximated on the same points
        :return: fitted function(s)
        """
        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercepts (bias)")
        if not (isinstance(f, list) or isinstance(f, Callable)):
            raise ValueError(f"f needs to be a function or a list of functions but is {type(f)}")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        self._self_implementation(y)

    def _approximate_sklearn(self, f: Union[Callable, List[Callable]]):
        """
        Approximates a (or multiple) function(s) with polynomials by least squares.
        :param f: function or list of functions that need to be approximated on the same points
        :return: fitted function(s)
        """
        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercepts (bias)")
        if not (isinstance(f, list) or isinstance(f, Callable)):
            raise ValueError(f"f needs to be a function or a list of functions but is {type(f)}")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        self._sklearn(y)

    def _approximate_nn(self, f: Callable):
        if isinstance(f, list):
            if len(f) == 1:
                f = f[0]
            else:
                raise ValueError(f"f needs to be a single function but is of type {type(f)}")

        y = f(self.grid.grid)

        dataset = LSDataset(self.basis, y, dim=self.grid.dim, scale=self.grid.scale)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        model = LSNN(input_dim=self.basis.shape[1], output_dim=1, weights=self.coeff)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
        criterion = torch.nn.MSELoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        writer = SummaryWriter(log_dir=os.path.join('logs', 'tensorboard'))

        train(model, criterion, optimizer, scheduler, dataloader, num_epochs=1000, device=device, writer=writer)

        coeff = model.state_dict()['linear.weight'].cpu().numpy().squeeze()

        self.coeff = coeff

    def _self_implementation(self, y: np.ndarray):

        if self.grid.grid_type == GridType.RANDOM_CHEBYSHEV:
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.grid):
                weight[i] = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(row) / np.pi))

        elif self.grid.grid_type == GridType.RANDOM_UNIFORM:
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported grid type {self.grid.grid_type}")

        print("Warning: The following is very unstable, since we calculate A.T@A")

        x_poly = (weight * self.basis.T).T
        self.basis = None
        y_prime = x_poly.T @ (weight * y.T).T
        x2 = x_poly.T @ x_poly
        del x_poly
        self.L, self.U = lu(x2, permute_l=True)[-2:]
        coeff = np.linalg.solve(self.U, np.linalg.solve(self.L, y_prime))
        self.coeff = coeff

    def _sklearn(self, y: np.ndarray):
        x_poly = self.basis
        model = LinearRegression()
        model.fit(x_poly, y)

        self.coeff = model.coef_.T

    def _rls(self, y: np.ndarray):
        x_poly = self.basis

        n = x_poly.shape[1]

        f = pa.filters.FilterRLS(n=n, mu=0.95, w='random')  # TODO: Initialize better if possible
        f.run(y, x_poly)

        self.coeff = f.w

    def _iterative_rls(self, y: np.ndarray):
        x_poly = self.basis

        n = x_poly.shape[1]

        n_samples = x_poly.shape[0]

        f = pa.filters.FilterRLS(n, mu=0.95, w='random')  # TODO: Initialize better if possible

        for k in range(n_samples):
            sample = x_poly[k, :]
            # y_hat_sample = f.predict(sample)
            y_sample = y[k]
            f.adapt(y_sample, sample)

        self.coeff = f.w

    def _approximate_lsmr(self, f: Callable):
        grid = self.grid.grid
        if not self.include_bias:  # TODO: Maybe remove this statement (everywhere)
            print("Please be aware that the result may become significantly worse when using no intercepts (bias)")
        if not (isinstance(f, list) or isinstance(f, Callable)):
            raise ValueError(f"f needs to be a function or a list of functions but is {type(f)}")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            print(
                "Warning: LSMR does not support multiple functions in parallel. "
                "So the program will run for each method from scratch again")
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        x_poly = self.basis
        if y.ndim == 1:
            res = lsmr(x_poly, y)
            coeffs = res[0]
        else:
            coeffs = np.empty((x_poly.shape[1], y.shape[1]))
            for i in range(y.shape[1]):
                res = lsmr(x_poly, y[:, i])
                coeffs[:, i] = res[0]
        self.coeff = coeffs

    def _approximate_pt(self, f: Union[Callable, list[Callable]], driver="gelss"):

        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)
        y = torch.tensor(y)
        x_poly = torch.tensor(self.basis)

        sol = torch.linalg.lstsq(x_poly, y, rcond=None, driver=driver)
        coeff = sol[0]

        self.coeff = coeff.cpu().numpy()

    def _approximate_numpy_lstsq(self, f: Union[Callable, list[Callable]]):

        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        # weighted least squares
        if self.grid.grid_type == GridType.RANDOM_CHEBYSHEV:
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.grid):
                weight[i] = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(row)))

        elif self.grid.grid_type == GridType.RANDOM_UNIFORM:
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported grid type {self.grid.grid_type}")

        x_poly = (weight * self.basis.T).T
        del self.basis
        y_prime = (weight * y.T).T

        sol = np.linalg.lstsq(x_poly, y_prime, rcond=None)
        coeff = sol[0]

        self.coeff = coeff

    def _approximate_rls(self, f: Callable):

        grid = self.grid.grid
        if not (isinstance(f, list) or isinstance(f, Callable)):
            raise ValueError(f"f needs to be a function or a list of functions but is {type(f)}")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            print(
                "Warning: (Iterative) RLS does not support multiple functions in parallel. "
                "So the program will run for each method from scratch again")
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        x_poly = self.basis
        if y.ndim == 1:
            self._rls(y)
        else:
            coeffs = np.empty((x_poly.shape[1], y.shape[1]))
            for i in range(y.shape[1]):
                self._rls(y[:, i])
                coeffs[:, i] = self.coeff
            self.coeff = coeffs

    def _approximate_iterative_rls(self, f: Callable):
        self._approximate_rls(f)
