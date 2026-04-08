"""
Tests for FaberBasisGenerator (Smolyak-structured hat basis).

Numerical tests run unconditionally.
Visualisation tests are tagged ``visual`` and are skipped by default::

    # run only numerical tests (default)
    pytest tests/test_faber_basis.py

    # include visualisation tests (saves PNGs to tests/figures/faber/)
    pytest tests/test_faber_basis.py -m visual
"""
import os
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt

import numpy as np
import pytest

from basis.faber import FaberBasisGenerator
from grid.grid.grid import Grid
from grid.rule.random_grid_rule import RandomGridRule
from utils.utils import calculate_num_points

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "faber")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Minimal test grid
# ---------------------------------------------------------------------------

class _TestGrid(Grid):
    """Lightweight Grid stub for unit-tests."""

    def __init__(self, pts: np.ndarray, scale: int,
                 lower: float = 0., upper: float = 1.):
        super().__init__(
            input_dim=pts.shape[1],
            scale=scale,
            grid=pts,
            rule=RandomGridRule.UNIFORM,
            lower_bound=lower,
            upper_bound=upper,
        )

    def get_num_points(self) -> int:
        return self.grid.shape[0]

    def __array__(self, dtype=None):
        return self.grid


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gen():
    return FaberBasisGenerator()


# ---------------------------------------------------------------------------
# Basis count tests – must match calculate_num_points
# ---------------------------------------------------------------------------

class TestBasisCount:
    @pytest.mark.parametrize("dim,scale", [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2),
        (5, 1), (5, 2),
    ])
    def test_n_basis_matches_smolyak(self, dim, scale):
        """Number of basis columns equals calculate_num_points(dim, scale)."""
        expected = calculate_num_points(dim, scale)
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, (max(10, expected), dim))
        V = np.array(_gen().create_basis(_TestGrid(pts, scale=scale)))
        assert V.shape[1] == expected, (
            f"dim={dim}, scale={scale}: got {V.shape[1]} columns, "
            f"expected {expected}"
        )


# ---------------------------------------------------------------------------
# 1-D tests
# ---------------------------------------------------------------------------

class Test1D:
    def test_vandermonde_shape_1d(self):
        """Vandermonde column count matches Smolyak count in 1-D."""
        pts = np.linspace(0, 1, 20).reshape(-1, 1)
        for scale in range(1, 5):
            expected = calculate_num_points(1, scale)
            B = _gen().create_basis(_TestGrid(pts, scale=scale))
            assert np.array(B).shape == (20, expected)

    def test_nonnegative_1d(self):
        """All basis values are non-negative (hat functions are ≥ 0)."""
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 1, (200, 1))
        for scale in [1, 2, 3, 4]:
            V = np.array(_gen().create_basis(_TestGrid(pts, scale=scale)))
            assert (V >= -1e-15).all(), f"Negative values at scale={scale}"

    def test_hat_params_consistency(self):
        """_build_hat_params returns m_i(n) entries."""
        for n in range(1, 8):
            knots, spacings = FaberBasisGenerator._build_hat_params(n, 0.0, 1.0)
            expected = FaberBasisGenerator._m_i(n)
            assert len(knots) == expected, f"n={n}: got {len(knots)}, expected {expected}"
            assert len(spacings) == expected

    def test_phi_chain_cardinality(self):
        """_phi_chain(n) total indices equals m_i(n)."""
        for n in range(1, 8):
            chain = FaberBasisGenerator._phi_chain(n)
            total = sum(len(v) if hasattr(v, '__len__') else len(list(v)) for v in chain.values())
            expected = FaberBasisGenerator._m_i(n)
            assert total == expected, f"n={n}: chain total {total} != m_i({n})={expected}"


# ---------------------------------------------------------------------------
# 2-D tests
# ---------------------------------------------------------------------------

class Test2D:
    def test_vandermonde_shape_2d(self):
        """Vandermonde shape in 2-D."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (30, 2))
        for scale in [1, 2, 3]:
            expected = calculate_num_points(2, scale)
            B = _gen().create_basis(_TestGrid(pts, scale=scale))
            assert np.array(B).shape == (30, expected)

    def test_nonnegative_2d(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(0, 1, (100, 2))
        V = np.array(_gen().create_basis(_TestGrid(pts, scale=2)))
        assert (V >= -1e-15).all()


# ---------------------------------------------------------------------------
# High-D smoke test
# ---------------------------------------------------------------------------

class TestHighDim:
    def test_shape_5d(self):
        """5-D with scale=2: n_basis = calculate_num_points(5, 2)."""
        rng = np.random.default_rng(1)
        expected = calculate_num_points(5, 2)
        pts = rng.uniform(0, 1, (max(10, expected), 5))
        V = np.array(_gen().create_basis(_TestGrid(pts, scale=2)))
        assert V.shape == (pts.shape[0], expected)

    def test_nonneg_5d(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(0, 1, (50, 5))
        V = np.array(_gen().create_basis(_TestGrid(pts, scale=1)))
        assert (V >= -1e-15).all()


# ---------------------------------------------------------------------------
# m_i tests
# ---------------------------------------------------------------------------

class TestMi:
    @pytest.mark.parametrize("i,expected", [
        (0, 0), (1, 1), (2, 3), (3, 5), (4, 9), (5, 17),
    ])
    def test_m_i(self, i, expected):
        assert FaberBasisGenerator._m_i(i) == expected

    def test_m_i_negative_raises(self):
        with pytest.raises(ValueError):
            FaberBasisGenerator._m_i(-1)


# ---------------------------------------------------------------------------
# Visualisation tests  (run with:  pytest -m visual)
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    """
    Each test saves one PNG to ``tests/figures/faber/``.
    Run with::

        pytest tests/test_faber_basis.py -m visual -v
    """

    def test_plot_1d_hats_by_scale(self):
        """Plot all 1-D hierarchical hat functions for several scales."""
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
        for ax, scale in zip(axes, [0, 1, 2, 3]):
            V = np.array(_gen().create_basis(_TestGrid(x, scale=scale)))
            for j in range(V.shape[1]):
                ax.plot(x.ravel(), V[:, j], lw=0.8)
            ax.set_title(f"scale={scale}  ({V.shape[1]} hats)")
            ax.set_xlim(0, 1)
        fig.suptitle("1-D Smolyak Hat Functions")
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "1d_hats_by_scale.png"), dpi=150)
        plt.close(fig)

