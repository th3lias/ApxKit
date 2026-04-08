import numpy as np
import scipy.sparse
from dataclasses import dataclass


@dataclass()
class Basis:
    """Container for a basis (Vandermonde) matrix, dense or sparse."""

    basis: np.ndarray | scipy.sparse.spmatrix

    def __array__(self, dtype=None, copy=None):
        if scipy.sparse.issparse(self.basis):
            arr = self.basis.toarray()
        else:
            arr = np.asarray(self.basis)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __matmul__(self, other):
        return self.basis @ other
