"""
Integration tests for Riesz Least Squares: fit → network → error comparison.

Checks:
  - LeastSquaresAlgorithm works with RieszBasisGenerator.
  - Coefficient shape matches 2·#I + 1.
  - Conversion to ReLU network preserves the approximation exactly.
  - Approximation error decreases as the index set grows (convergence).
  - Multi-function (vector-valued) fitting works.

Visualisation tests (tagged ``visual``) save PNGs to tests/figures/riesz_ls/.
Run with:  pytest tests/test_riesz_least_squares.py -m visual -v -s
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from algorithm.least_squares import LeastSquaresAlgorithm
from basis.riesz import RieszBasisGenerator
from basis.riesz_network import coefficients_to_network
from function.f import Function
from function.provider import ParametrizedFunctionProvider
from function.type import FunctionType
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.grid.random_grid import RandomGrid
from grid.rule.random_grid_rule import RandomGridRule
from solver.scipy_lstsq_solver import ScipyLstsqSolver
from utils.utils import l2_error_function_values, max_error_function_values

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "riesz_ls")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alg(max_freq: int, dim: int = 1,
              n_train: int = 200) -> LeastSquaresAlgorithm:
    """Create a Riesz LS algorithm for a 1-D index set up to max_freq."""
    if dim == 1:
        index_set = [(k,) for k in range(1, max_freq + 1)]
    else:
        # Simple axis-aligned index set for multi-D
        index_set = []
        for d in range(dim):
            for k in range(1, max_freq + 1):
                e = [0] * dim
                e[d] = k
                index_set.append(tuple(e))

    n_basis = 2 * len(index_set) + 1
    # Oversampling: need n_train > n_basis
    actual_n = max(n_train, 3 * n_basis)

    return LeastSquaresAlgorithm(
        basis_generator=RieszBasisGenerator(index_set),
        grid_generator=UniformGridGenerator(seed=42, multiplier_fun=lambda x: actual_n),
        solver=ScipyLstsqSolver(driver="gelsy"),
    )


def _eval_grid(dim: int, n: int = 500) -> RandomGrid:
    """Deterministic uniform evaluation grid."""
    rng = np.random.default_rng(999)
    pts = rng.uniform(0, 1, (n, dim))
    return RandomGrid(input_dim=dim, scale=1, n_points=n,
                      grid=pts, rule=RandomGridRule.UNIFORM)


def _l2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


_f_sine = Function(lambda x: np.sin(2 * np.pi * x[:, 0]), dim=1, name="sine")
_f_quad = Function(lambda x: x[:, 0] ** 2, dim=1, name="quad")


# ---------------------------------------------------------------------------
# Fit mechanics
# ---------------------------------------------------------------------------

class TestFitMechanics:
    def test_coeff_shape(self):
        """Coefficient vector has length 2·#I + 1."""
        alg = _make_alg(max_freq=5)
        alg.fit(dim=1, scale=1, f=_f_sine)
        assert alg.coeff.shape == (11, 1)   # 2*5 + 1

    def test_training_basis_freed(self):
        alg = _make_alg(max_freq=3)
        alg.fit(dim=1, scale=1, f=_f_sine)
        assert alg.basis is None

    def test_multi_function(self):
        """Fitting two functions simultaneously yields (n_basis, 2)."""
        alg = _make_alg(max_freq=4)
        alg.fit(dim=1, scale=1, f=[_f_sine, _f_quad])
        assert alg.coeff.shape == (9, 2)


# ---------------------------------------------------------------------------
# Approximation quality
# ---------------------------------------------------------------------------

class TestApproximationQuality:
    def test_convergence_1d(self):
        """L2 error on sin(2πx) decreases as max_freq grows."""
        errors = []
        for K in [2, 5, 10]:
            alg = _make_alg(max_freq=K)
            alg.fit(dim=1, scale=1, f=_f_sine)
            eg = _eval_grid(dim=1, n=500)
            y_hat = alg.evaluate(eg).ravel()
            y_true = np.sin(2 * np.pi * np.array(eg)[:, 0])
            errors.append(_l2(y_true, y_hat))

        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error did not decrease: K={[2,5,10][i]}→{[2,5,10][i+1]}: "
                f"{errors[i]:.4e} → {errors[i+1]:.4e}"
            )


# ---------------------------------------------------------------------------
# Network conversion fidelity
# ---------------------------------------------------------------------------

class TestNetworkConversion:
    @pytest.mark.parametrize("max_freq", [3, 5, 8])
    def test_conversion_error_at_machine_eps(self, max_freq):
        """LS coefficients → ReLU network introduces zero approximation error."""
        alg = _make_alg(max_freq=max_freq)
        alg.fit(dim=1, scale=1, f=_f_sine)

        gen = alg.basis_generator
        alpha_0, coeff_dict = gen.coefficients_to_dict(alg.coeff)
        net = coefficients_to_network(alpha_0, coeff_dict, dim=1)

        eg = _eval_grid(dim=1, n=1000)
        y_ls = alg.evaluate(eg).ravel()
        y_net = net(np.array(eg))

        conv_err = max_error_function_values(y_ls, y_net)
        assert conv_err < 1e-12, f"Conversion error {conv_err:.2e} exceeds tolerance"

    def test_conversion_preserves_error_2d(self):
        """2-D case: network error matches LS error."""
        alg = _make_alg(max_freq=3, dim=2, n_train=400)
        f_2d = Function(lambda x: np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]),
                        dim=2, name="sin_cos")
        alg.fit(dim=2, scale=1, f=f_2d)

        gen = alg.basis_generator
        alpha_0, coeff_dict = gen.coefficients_to_dict(alg.coeff)
        net = coefficients_to_network(alpha_0, coeff_dict, dim=2)

        eg = _eval_grid(dim=2, n=500)
        pts = np.array(eg)
        y_true = f_2d(pts)
        y_ls = alg.evaluate(eg).ravel()
        y_net = net(pts)

        ls_err = l2_error_function_values(y_true, y_ls)
        net_err = l2_error_function_values(y_true, y_net)
        conv_err = max_error_function_values(y_ls, y_net)

        assert abs(ls_err - net_err) < 1e-10, (
            f"LS error {ls_err:.6e} ≠ net error {net_err:.6e}"
        )
        assert conv_err < 1e-12


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    def test_plot_end_to_end(self):
        """3×3 grid: function → LS fit → ReLU network → error comparison."""
        rng = np.random.default_rng(42)
        x_plot = np.linspace(0, 1, 500).reshape(-1, 1)
        max_freq = 8

        configs = [
            (FunctionType.GAUSSIAN,    1.0, "Gaussian"),
            (FunctionType.CONTINUOUS,  1.0, "Continuous"),
            (FunctionType.CORNER_PEAK, 1.0, "Corner Peak"),
        ]

        fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4 * len(configs)))

        for row, (ftype, avg_c, label) in enumerate(configs):
            c = avg_c * np.ones(1) + 0.1 * rng.standard_normal(1)
            w = rng.uniform(0, 1, size=1)
            f = ParametrizedFunctionProvider.get_function(ftype, d=1, c=c, w=w)

            alg = _make_alg(max_freq=max_freq, n_train=200)
            alg.fit(dim=1, scale=1, f=f)

            gen = alg.basis_generator
            alpha_0, coeff_dict = gen.coefficients_to_dict(alg.coeff)
            net = coefficients_to_network(alpha_0, coeff_dict, dim=1)

            plot_grid = RandomGrid(input_dim=1, scale=1, n_points=500,
                                   grid=x_plot, rule=RandomGridRule.UNIFORM)

            y_true = f(x_plot)
            y_ls = alg.evaluate(plot_grid).ravel()
            y_net = net(x_plot)

            # Column 0: functions
            ax = axes[row, 0]
            ax.plot(x_plot, y_true, "k-", lw=1.5, label="True function")
            ax.plot(x_plot, y_ls, "b--", lw=1.5, label="Riesz LS fit")
            ax.plot(x_plot, y_net, "r:", lw=1.5, label="ReLU network")
            ax.set_title(label)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Column 1: approximation error
            ax = axes[row, 1]
            ax.semilogy(x_plot, np.maximum(np.abs(y_true - y_ls), 1e-17),
                        "b--", lw=1.2, label="LS error")
            ax.semilogy(x_plot, np.maximum(np.abs(y_true - y_net), 1e-17),
                        "r:", lw=1.2, label="Net error")
            ax.set_title("Approximation error")
            ax.set_xlabel("$x$")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Column 2: conversion error LS ↔ network
            ax = axes[row, 2]
            conv_err = np.abs(y_ls - y_net)
            ax.semilogy(x_plot, np.maximum(conv_err, 1e-17), "purple", lw=1.2)
            ax.axhline(np.finfo(float).eps, color="red", ls=":", alpha=0.5,
                        label=f"machine $\\varepsilon$")
            ax.set_title("Conversion error (LS $\\leftrightarrow$ net)")
            ax.set_xlabel("$x$")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Riesz LS → ReLU Network  (d=1, K={max_freq}, "
            f"$N_{{train}}$=200)",
            fontsize=14, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "end_to_end.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
