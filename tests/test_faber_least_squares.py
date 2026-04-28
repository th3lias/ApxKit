"""
Integration tests for Faber (hat-function) Least Squares using LeastSquaresAlgorithm.

Checks:
  - Training and evaluation run without error.
  - Approximation error decreases as scale increases (convergence).
  - The algorithm approximates simple targets well.
  - Multi-function (vector-valued) fitting works.
  - Coefficient shape matches calculate_num_points(dim, scale).

Visualisation tests (tagged ``visual``) save PNGs to tests/figures/faber_ls/.
Run with:  pytest tests/test_faber_least_squares.py -m visual -v -s
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from algorithm.least_squares import LeastSquaresAlgorithm
from basis.faber import FaberBasisGenerator
from function.f import Function
from grid.grid.random_grid import RandomGrid
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.rule.random_grid_rule import RandomGridRule
from solver.scipy_lstsq_solver import ScipyLstsqSolver
from utils.utils import calculate_num_points

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "faber_ls")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alg(multiplier=2) -> LeastSquaresAlgorithm:
    """Create a Faber LS algorithm with the given oversampling multiplier."""
    return LeastSquaresAlgorithm(
        basis_generator=FaberBasisGenerator(),
        grid_generator=UniformGridGenerator(seed=42, multiplier_fun=lambda x: multiplier * x),
        solver=ScipyLstsqSolver(driver='gelsy'),
    )


def _eval_grid(dim: int, scale: int, n: int = 300,
               lower: float = 0., upper: float = 1.) -> RandomGrid:
    """Deterministic uniform evaluation grid with correct scale for basis."""
    rng = np.random.default_rng(999)
    pts = rng.uniform(lower, upper, (n, dim))
    return RandomGrid(
        input_dim=dim, scale=scale, n_points=n,
        grid=pts, rule=RandomGridRule.UNIFORM,
        lower_bound=lower, upper_bound=upper,
    )


def _l2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


# Simple test functions
_f_linear  = Function(lambda x: x[:, 0],                      dim=1, name="linear")
_f_quad    = Function(lambda x: x[:, 0] ** 2,                 dim=1, name="quad")
_f_sine    = Function(lambda x: np.sin(2 * np.pi * x[:, 0]),  dim=1, name="sine")
_f_2d_sum  = Function(lambda x: x[:, 0] + x[:, 1],            dim=2, name="2d_sum")
_f_2d_prod = Function(lambda x: x[:, 0] * x[:, 1],            dim=2, name="2d_prod")


# ---------------------------------------------------------------------------
# Fit mechanics
# ---------------------------------------------------------------------------

class TestFitMechanics:
    def test_coeff_shape_1d(self):
        """Coefficient vector has length calculate_num_points(1, scale)."""
        alg = _make_alg()
        alg.fit(dim=1, scale=2, f=_f_linear)
        n_basis = calculate_num_points(1, 2)
        assert alg.coeff.shape == (n_basis, 1)

    def test_coeff_shape_2d(self):
        alg = _make_alg()
        alg.fit(dim=2, scale=2, f=_f_2d_sum)
        n_basis = calculate_num_points(2, 2)
        assert alg.coeff.shape == (n_basis, 1)

    def test_training_basis_freed_after_fit(self):
        """The training Vandermonde matrix must be freed to save memory."""
        alg = _make_alg()
        alg.fit(dim=1, scale=2, f=_f_linear)
        assert alg.basis is None

    def test_multi_function_coeff_shape(self):
        """Fitting two functions simultaneously yields (n_basis, 2) coefficients."""
        alg = _make_alg()
        alg.fit(dim=1, scale=2, f=[_f_linear, _f_quad])
        n_basis = calculate_num_points(1, 2)
        assert alg.coeff.shape == (n_basis, 2)


# ---------------------------------------------------------------------------
# Approximation quality
# ---------------------------------------------------------------------------

class TestApproximationQuality:
    def test_linear_1d(self):
        """
        Linear f(x)=x should be well-approximated at moderate scale.
        """
        alg = _make_alg(multiplier=4)
        alg.fit(dim=1, scale=3, f=_f_linear)
        eg = _eval_grid(dim=1, scale=3, n=500)
        y_hat = alg.evaluate(eg).ravel()
        y_true = np.array(eg)[:, 0]
        assert _l2(y_true, y_hat) < 0.05

    def test_convergence_1d_sine(self):
        """L2 error on sin(2πx) decreases as scale grows."""
        errors = []
        for scale in [2, 4, 6]:
            alg = _make_alg(multiplier=4)
            alg.fit(dim=1, scale=scale, f=_f_sine)
            eg = _eval_grid(dim=1, scale=scale, n=500)
            pts = np.array(eg)
            y_hat = alg.evaluate(eg).ravel()
            y_true = np.sin(2 * np.pi * pts[:, 0])
            errors.append(_l2(y_true, y_hat))

        # each level should be strictly better
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error did not decrease: scale {[2,4,6][i]}→{[2,4,6][i+1]}: "
                f"{errors[i]:.4e} → {errors[i+1]:.4e}"
            )

    def test_convergence_2d(self):
        """L2 error on a smooth 2-D function decreases as scale grows."""
        f_2d_trig = Function(
            lambda x: np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]),
            dim=2, name="sin_cos",
        )
        errors = []
        for scale in [2, 4, 6]:
            alg = _make_alg(multiplier=4)
            alg.fit(dim=2, scale=scale, f=f_2d_trig)
            eg = _eval_grid(dim=2, scale=scale, n=400)
            pts = np.array(eg)
            y_hat = alg.evaluate(eg).ravel()
            y_true = np.sin(2 * np.pi * pts[:, 0]) * np.cos(2 * np.pi * pts[:, 1])
            errors.append(_l2(y_true, y_hat))

        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error did not decrease: scale {[2,4,6][i]}→{[2,4,6][i+1]}: "
                f"{errors[i]:.4e} → {errors[i+1]:.4e}"
            )


# ---------------------------------------------------------------------------
# Visualisation tests  (run with:  pytest -m visual)
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    def test_plot_1d_convergence(self):
        """Plot the approximation at increasing scales."""
        x_plot = np.linspace(0, 1, 500).reshape(-1, 1)
        y_true = np.sin(2 * np.pi * x_plot[:, 0])

        fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
        for ax, scale in zip(axes, [1, 2, 4, 6]):
            alg = _make_alg(multiplier=4)
            alg.fit(dim=1, scale=scale, f=_f_sine)
            eg = RandomGrid(
                input_dim=1, scale=scale, n_points=500,
                grid=x_plot, rule=RandomGridRule.UNIFORM,
                lower_bound=0., upper_bound=1.,
            )
            y_hat = alg.evaluate(eg).ravel()
            ax.plot(x_plot, y_true, 'k-', lw=1, label='true')
            ax.plot(x_plot, y_hat, 'r--', lw=1, label='approx')
            n_basis = calculate_num_points(1, scale)
            ax.set_title(f"scale={scale} ({n_basis} basis)")
            ax.legend(fontsize=7)
        fig.suptitle("Faber LS: sin(2πx) convergence")
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "1d_sine_convergence.png"), dpi=150)
        plt.close(fig)

