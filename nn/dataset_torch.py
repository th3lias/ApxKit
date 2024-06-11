from functools import reduce
from itertools import product
from typing import Union, List, Tuple
from operator import mul

import numpy as np
from torch.utils.data import Dataset

from interpolate.partition import Partition
from utils.utils import load_basis_indices_if_existent, save_basis_indices


# write Dataset that takes grid from GridProvider

class LS_Dataset(Dataset):
    def __init__(self, grid, y, dim: int, scale: int):
        self.grid = grid
        self.y = y
        self.dim = dim
        self.scale = scale
        self._b_idx = None
        # self._b_idx = load_basis_indices_if_existent(self.dim, self.scale)

    def __len__(self):
        return self.grid.shape[0]
        return self.grid.get_num_points()

    def __getitem__(self, idx):
        # x = self._make_basis(self.grid.get_grid()[idx])
        x = self.grid[idx]
        y = self.y[idx]
        return (x, y)

    def _make_basis(self, x):

        if self._b_idx is None:
            self._idx = self._smolyak_idx()
            self._b_idx = self._poly_idx(self._idx)
            save_basis_indices(self._b_idx, self.dim, self.scale)

        scale = self.scale

        grid = self.grid.grid

        ts = self._cheby2n(x.T, self._m_i(scale + 1)).reshape(-1, self.dim, 1)
        n_polys = len(self._b_idx)
        npts = grid.shape[0]
        basis = np.empty(shape=(npts, n_polys))

        for ind, comb in enumerate(self._b_idx):
            basis[:, ind] = reduce(mul, [ts[comb[i] - 1, i, :] for i in range(self.dim)])

        return basis

    # TODO: Same method like in interpolator.py
    def _smolyak_idx(self):
        scale = self.scale
        dim = self.dim
        if not isinstance(scale, int):
            raise ValueError(f"Scale must have an int type but is {type(scale)}")

        idx_list = list()
        for q in range(dim, scale + dim + 1):
            p = Partition(dim, q, limit=1)
            idx_list.extend(p.get_all_partitions())

        return idx_list

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
        elif i < 2:
            return i
        else:
            return 2 ** (i - 1) + 1

    @staticmethod
    def _cheby2n(x, n):
        """
        Computes the first :math:`n+1` Chebyshev polynomials of the first
        kind evaluated at each point in :math:`x` .
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
