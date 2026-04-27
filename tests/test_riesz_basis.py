"""
Tests for the Riesz basis functions and RieszBasisGenerator.

Checks:
  - Known values of C(t) and S(t) at special points.
  - Periodicity and interpolation of cos/sin at {0, 1/4, 1/2, 3/4, 1}.
  - Vandermonde matrix shape from RieszBasisGenerator.
  - Column content matches direct evaluation of C_k, S_k.

Visualisation tests (tagged ``visual``) save PNGs to tests/figures/riesz/.
Run with:  pytest tests/test_riesz_basis.py -m visual -v -s
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from basis.riesz import eval_C, eval_S, eval_C_k, eval_S_k, RieszBasisGenerator
from grid.grid.random_grid import RandomGrid
from grid.rule.random_grid_rule import RandomGridRule

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "riesz")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Basis function sanity checks
# ---------------------------------------------------------------------------

class TestBasisFunctions:
    def test_C_known_values(self):
        assert eval_C(0.0) == pytest.approx(1.0)
        assert eval_C(0.25) == pytest.approx(0.0)
        assert eval_C(0.5) == pytest.approx(-1.0)
        assert eval_C(0.75) == pytest.approx(0.0)
        assert eval_C(1.0) == pytest.approx(1.0)

    def test_S_known_values(self):
        assert eval_S(0.0) == pytest.approx(0.0)
        assert eval_S(0.25) == pytest.approx(1.0)
        assert eval_S(0.5) == pytest.approx(0.0)
        assert eval_S(0.75) == pytest.approx(-1.0)
        assert eval_S(1.0) == pytest.approx(0.0)

    def test_C_periodic(self):
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose(eval_C(t), eval_C(t + 3.0), atol=1e-12)

    def test_S_periodic(self):
        t = np.linspace(0, 1, 50)
        np.testing.assert_allclose(eval_S(t), eval_S(t + 5.0), atol=1e-12)

    def test_C_interpolates_cos(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(eval_C(t), np.cos(2 * np.pi * t), atol=1e-12)

    def test_S_interpolates_sin(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(eval_S(t), np.sin(2 * np.pi * t), atol=1e-12)


# ---------------------------------------------------------------------------
# Multivariate helpers
# ---------------------------------------------------------------------------

class TestMultivariate:
    def test_C_k_1d(self):
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        k = np.array([3.0])
        np.testing.assert_allclose(eval_C_k(x, k), eval_C(3.0 * x[:, 0]), atol=1e-12)

    def test_S_k_2d(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, (200, 2))
        k = np.array([1.0, -1.0])
        np.testing.assert_allclose(eval_S_k(x, k), eval_S(x @ k), atol=1e-12)


# ---------------------------------------------------------------------------
# RieszBasisGenerator
# ---------------------------------------------------------------------------

class TestRieszBasisGenerator:
    def test_vandermonde_shape_1d(self):
        index_set = [(1,), (2,), (3,)]
        gen = RieszBasisGenerator(index_set)
        pts = np.linspace(0, 1, 50).reshape(-1, 1)
        grid = RandomGrid(input_dim=1, scale=1, n_points=50,
                          grid=pts, rule=RandomGridRule.UNIFORM)
        basis = gen.create_basis(grid)
        assert np.array(basis).shape == (50, 7)   # 2*3 + 1

    def test_vandermonde_shape_2d(self):
        index_set = [(1, 0), (0, 1), (1, 1)]
        gen = RieszBasisGenerator(index_set)
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (100, 2))
        grid = RandomGrid(input_dim=2, scale=1, n_points=100,
                          grid=pts, rule=RandomGridRule.UNIFORM)
        basis = gen.create_basis(grid)
        assert np.array(basis).shape == (100, 7)

    def test_constant_column(self):
        """First column of Vandermonde is all ones."""
        gen = RieszBasisGenerator([(1,)])
        pts = np.linspace(0, 1, 20).reshape(-1, 1)
        grid = RandomGrid(input_dim=1, scale=1, n_points=20,
                          grid=pts, rule=RandomGridRule.UNIFORM)
        V = np.array(gen.create_basis(grid))
        np.testing.assert_allclose(V[:, 0], 1.0)

    def test_columns_match_eval(self):
        """Vandermonde columns match direct eval_C_k / eval_S_k."""
        index_set = [(2,), (5,)]
        gen = RieszBasisGenerator(index_set)
        pts = np.linspace(0, 1, 100).reshape(-1, 1)
        grid = RandomGrid(input_dim=1, scale=1, n_points=100,
                          grid=pts, rule=RandomGridRule.UNIFORM)
        V = np.array(gen.create_basis(grid))

        np.testing.assert_allclose(V[:, 1], eval_C(2.0 * pts[:, 0]), atol=1e-12)
        np.testing.assert_allclose(V[:, 2], eval_S(2.0 * pts[:, 0]), atol=1e-12)
        np.testing.assert_allclose(V[:, 3], eval_C(5.0 * pts[:, 0]), atol=1e-12)
        np.testing.assert_allclose(V[:, 4], eval_S(5.0 * pts[:, 0]), atol=1e-12)

    def test_n_basis_property(self):
        gen = RieszBasisGenerator([(1,), (2,), (3,)])
        assert gen.n_basis == 7

    def test_coefficients_to_dict_roundtrip(self):
        """coefficients_to_dict extracts alpha_0 and pairs correctly."""
        index_set = [(1,), (2,)]
        gen = RieszBasisGenerator(index_set)
        beta = np.array([1.5, 0.3, -0.2, 0.0, 0.7])
        alpha_0, d = gen.coefficients_to_dict(beta)
        assert alpha_0 == pytest.approx(1.5)
        assert d[(1,)] == pytest.approx((0.3, -0.2))
        assert d[(2,)] == pytest.approx((0.0, 0.7))


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    def test_plot_basis_functions(self):
        """Plot C_k and S_k for k = 1, 2, 3."""
        x = np.linspace(0, 1, 500)
        fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)

        for col, k in enumerate([1, 2, 3]):
            axes[0, col].plot(x, eval_C(k * x), "b-", lw=1.2)
            axes[0, col].set_title(f"$\\mathcal{{C}}_{k}(x)$")
            axes[0, col].grid(True, alpha=0.3)

            axes[1, col].plot(x, eval_S(k * x), "r-", lw=1.2)
            axes[1, col].set_title(f"$\\mathcal{{S}}_{k}(x)$")
            axes[1, col].grid(True, alpha=0.3)

        fig.suptitle("Riesz basis functions $\\mathcal{C}_k$ and $\\mathcal{S}_k$")
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "basis_functions.png"), dpi=150)
        plt.close(fig)
