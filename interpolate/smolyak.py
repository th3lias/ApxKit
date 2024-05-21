from functools import reduce
from itertools import chain, combinations_with_replacement, product, permutations
from operator import mul
from typing import Callable, Union, List, Tuple, Generator

import numpy as np
from scipy.linalg import lu

from grid.grid_provider import GridProvider, GridType
from interpolate.interpolator import Interpolator


# most of the content is adapted from https://github.com/EconForge/Smolyak, which implemented the Smolyak algorithm
# based on the paper:

# Smolyak method for solving dynamic economic models:
# Lagrange interpolation, anisotropic grid and adaptive domain

# from Kenneth L. Judd, Lilia Maliar, Serguei Maliar and Rafael Valero


class SmolyakInterpolator(Interpolator):
    def __init__(self, dimension: int, scale: int, seed: int = None):
        self.dim = dimension
        self.scale = scale
        self.gp = GridProvider(self.dim, seed=seed)
        super().__init__(self.gp.generate(GridType.CHEBYSHEV, self.scale))

    def interpolate(self, f: Callable) -> Callable:
        basis = self._build_basis()
        l, u = lu(basis, permute_l=True)[-2:]
        coeff = np.linalg.solve(u, np.linalg.solve(l, f(self.grid.grid)))

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_smolyak = self._build_basis(grid=data, b_idx=self._b_idx)
            return data_smolyak @ coeff

        return f_hat

    def _poly_idx(self, idx: Union[List[List[int]], None] = None) -> List[Tuple[int]]:
        """
        Build indices specifying all the Cartesian products of Chebyshev
        polynomials needed to build Smolyak polynomial
        Parameters
        ----------
        idx : list (list (int)), optional (default=None)
            The Smolyak indices for parameters dim and scale. Should be computed
            by calling `smol_idx(dim, scale)`. If None is given, the indices
            are computed using this function call
        Returns
        -------
        phi_idx : array : (int, ndim=2)
            A two-dimensional array of integers where each row specifies a
            new set of indices needed to define a Smolyak basis polynomial
        Notes
        -----
        This function uses smol_idx and phi_chain. The output of this
        function is used by build_B to construct the B matrix
        """
        scale = self.scale
        if idx is None:
            idx = self._smolyak_idx()
        if not isinstance(scale, int):
            raise ValueError(f"Scale must have an int type but is {type(scale)}")
        aphi = self._phi_chain(scale + 1)
        base_polys = []
        for el in idx:
            temp = [aphi[i] for i in el]
            # Save these indices that we iterate through because
            # we need them for the Chebyshev polynomial combination
            # idx.append(el)
            base_polys.extend(list(product(*temp)))
        return base_polys

    def _smolyak_idx(self) -> List[List[int]]:
        """
        Fidx all the indices that satisfy the requirement that
        math::
        d \\leq \\sum_{i=1}^d \\leq d + scale.
        Returns
        -------
        true_idx : array
            A 1-d Any array containing all d element arrays satisfying the
            constraint
        Notes
        -----
        This function is used directly by build_grid and poly_idx
        """
        scale = self.scale
        dim = self.dim
        if not isinstance(scale, int):
            raise ValueError(f"Scale must have an int type but is {type(scale)}")
        # Need to capture up to value scale + 1 so in python need scale+2
        possible_values = range(1, scale + 2)
        # find all (i1, i2, ... id) such that their sum is in range
        # we want; this will cut down on later iterations
        poss_idx = [el for el in combinations_with_replacement(possible_values, dim) if dim < sum(el) <= dim + scale]
        true_idx = [[el for el in self._permute(list(val))] for val in poss_idx]
        # Add the d dimension 1 array so that we don't repeat it a bunch
        # of times
        true_idx.extend([[[1] * dim]])
        t_idx = list(chain.from_iterable(true_idx))
        return t_idx

    def _build_basis(self, grid: np.ndarray = None, b_idx: Union[List[Tuple[int]], None] = None):
        if b_idx is None:
            self._idx = self._smolyak_idx()
            self._b_idx = self._poly_idx(self._idx)
        else:
            self._b_idx = b_idx
        scale = self.scale

        if grid is None:
            grid = self.grid.grid

        ts = self._cheby2n(x=grid.T, n=self._m_i(scale + 1))
        n_polys = len(self._b_idx)
        npts = grid.shape[0]
        basis = np.empty((npts, n_polys), order='F')
        for ind, comb in enumerate(self._b_idx):
            # multiplying the polynomials for each dimension
            basis[:, ind] = reduce(mul, [ts[comb[i] - 1, i, :] for i in range(self.dim)])
        return basis

    @staticmethod
    def _phi_chain(n):
        """
        For each number in 1 to n, compute the Smolyak indices for the
        corresponding basis functions. This is the :math:`n` in
        :math:`\\phi_n`
        Parameters
        ----------
        n : int
            The last Smolyak index :math:`n` for which the basis polynomial
            indices should be found
        Returns
        -------
        aphi_chain : dict (int -> list)
            A dictionary whose keys are the Smolyak index :math:`n` and
            values are lists containing all basis polynomial subscripts for
            that Smolyak index
        """
        aphi_chain = dict()
        aphi_chain[1] = [1]
        aphi_chain[2] = [2, 3]
        curr_val = 4
        for i in range(3, n + 1):
            end_val = 2 ** (i - 1) + 1
            temp = range(curr_val, end_val + 1)
            aphi_chain[i] = temp
            curr_val = end_val + 1
        return aphi_chain

    @staticmethod
    def _m_i(i: int):
        r"""
        Compute one plus the "total degree of the interpolating
        polynomials" (Kruger & Kubler, 2004). This shows up many times in
        Smolyak's algorithm. It is defined as:
        math::
            m_i = \begin{cases}
            1 \quad & \text{if } i = 1 \\
            2^{i-1} + 1 \quad & \text{if } i \geq 2
            \end{cases}
        Parameters
        ----------
        i : int
            The integer i which the total degree should be evaluated
        Returns
        -------
        num : int
            Return the value given by the expression above
        """
        if i < 0:
            raise ValueError('i must be positive')
        elif i == 0:
            return 0
        elif i == 1:
            return 1
        else:
            return 2 ** (i - 1) + 1

    @staticmethod
    def _permute(array: Union[list, np.ndarray], drop_duplicates: bool = True) -> Generator:
        """
        Creates a generator object that yields all permutations of the given array/list. The permutations are unique,
        if the parameter drop_duplicates is set to True
        At the beginning, the array/list gets sorted.
        :param array: Array or List where the permutations should be calculated
        :param drop_duplicates: If True, a permutation which is the same as another permutation since there were
        duplicate values in the array is dropped, otherwise it is kept
        """
        if isinstance(array, np.ndarray):
            if array.ndim == 1:
                array = np.sort(array)
            else:
                raise ValueError(
                    f"Wrong number of dimensions for the parameter 'array'. Expected ndim=1 but got {array.ndim}"
                )
        elif isinstance(array, list):
            array = sorted(array)
        else:
            raise ValueError(f"Expected 'array' to be a list or a np.ndarray but got {type(array)}")

        seen = set()

        for perm in permutations(array):
            if perm not in seen or not drop_duplicates:
                seen.add(perm)
                yield list(perm)

    @staticmethod
    def _cheby2n(x, n):
        """
        Computes the first :math:`n+1` Chebyshev polynomials of the first
        kind evaluated at each point in :math:`x` .
        Note that we can calculate the n+1-st Chebyshev polynomial by
        T_{n+1} = 2 x T_n - T_{n-1}
        Parameters
        ----------
        x : float or array(float)
            A single point (float) or an array of points where each
            polynomial should be evaluated
        n : int
            The integer specifying which Chebyshev polynomial is the last
            to be computed
        Returns
        -------
        results : array (float, ndim=x.ndim+1)
            The results of computation. This will be an :math:`(n+1 \\times
            dim \\dots)` where :math:`(dim \\dots)` is the shape of x. Each
            slice along the first dimension represents a new Chebyshev
            polynomial. This dimension has length :math:`n+1` because it
            includes :math:`\\phi_0` which is equal to 1 :math:`\\forall x`
        """
        x = np.asarray(x)
        dim = x.shape
        results = np.zeros((n + 1,) + dim)
        results[0, ...] = np.ones(dim)
        results[1, ...] = x
        for i in range(2, n + 1):
            results[i, ...] = 2 * x * results[i - 1, ...] - results[i - 2, ...]
        return results
