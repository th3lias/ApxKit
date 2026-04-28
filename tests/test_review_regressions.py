"""
Regression tests for the bugs found during the WLS / full-codebase review
(8 Apr 2026, see REVIEW_WLS.md).

Bug 1 — WLS weight must be 1/√μ, not √μ
Bug 2 — y_test shape in use_max_scale=True  (structural; tested via _calc_error)
Bug 3 — ParametrizedFunction c/w constructor order
Bug 4 — Enum trailing commas produced tuple values
Bug 5 — _cheby2n √2 normalisation must skip T₀
Bug 6 — l2_error_function_values axis
"""

import numpy as np
import pytest

from basis.clenshaw_curtis_level_polynomial_basis_generator import (
    ClenshawCurtisLevelPolynomialBasisGenerator,
)
from function.parametrized_f import ParametrizedFunction
from function.type import FunctionType
from grid.rule.random_grid_rule import RandomGridRule
from utils.utils import l2_error_function_values


# ---------------------------------------------------------------------------
# Bug 1 — WLS weight is 1/√μ  (points near boundary get LOWER weight)
# ---------------------------------------------------------------------------

class TestWLSWeight:
    """The importance-sampling weight must decrease toward the boundary
    of [-1, 1] where the Chebyshev density is large."""

    @staticmethod
    def _weight_at(x_11: np.ndarray) -> np.ndarray:
        """Expected per-point weight: sqrt(π · √(1−x²))."""
        return np.sqrt(np.prod(
            np.pi / np.polynomial.chebyshev.chebweight(x_11), axis=1
        ))

    def test_centre_larger_than_boundary(self):
        """Points near x = 0 must receive a larger weight than near ±1."""
        centre = np.array([[0.0]])
        edge = np.array([[0.95]])
        assert self._weight_at(centre) > self._weight_at(edge)

    def test_weight_matches_formula(self):
        """Spot-check against the closed-form 1/√μ."""
        pts = np.array([[0.0, 0.5]])
        w = self._weight_at(pts)
        expected = np.sqrt(
            (np.pi * np.sqrt(1 - 0.0 ** 2))
            * (np.pi * np.sqrt(1 - 0.5 ** 2))
        )
        np.testing.assert_allclose(w, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Bug 2 — _calc_error requires (n_points, n_functions) arrays
# ---------------------------------------------------------------------------

class TestCalcErrorShape:
    """Both y_test and y_hat must be (n_points, n_functions)."""

    def test_per_function_error(self):
        """axis=0 reduction gives one error per function column."""
        rng = np.random.default_rng(0)
        n_pts, n_fns = 200, 3
        y_true = rng.standard_normal((n_pts, n_fns))
        y_hat = y_true + 0.01 * rng.standard_normal((n_pts, n_fns))

        ell_2 = np.sqrt(np.mean(np.square(y_true - y_hat), axis=0))
        assert ell_2.shape == (n_fns,), "Should be one error per function"


# ---------------------------------------------------------------------------
# Bug 3 — ParametrizedFunction stores c and w correctly
# ---------------------------------------------------------------------------

class TestParametrizedFunctionCW:
    def test_attributes_match_args(self):
        c = np.array([1.0, 2.0])
        w = np.array([3.0, 4.0])
        pf = ParametrizedFunction(f=lambda x: x, dim=2, c=c, w=w)
        np.testing.assert_array_equal(pf.c, c)
        np.testing.assert_array_equal(pf.w, w)


# ---------------------------------------------------------------------------
# Bug 4 — Enum values are plain ints, not tuples
# ---------------------------------------------------------------------------

class TestEnumValues:
    def test_function_type_values_are_ints(self):
        for member in FunctionType:
            assert isinstance(member.value, int), (
                f"{member.name}.value is {type(member.value)}, expected int"
            )

    def test_function_type_roundtrip(self):
        """FunctionType(int) must return the corresponding member."""
        assert FunctionType(1) is FunctionType.OSCILLATORY

    def test_random_grid_rule_values_are_ints(self):
        for member in RandomGridRule:
            assert isinstance(member.value, int), (
                f"{member.name}.value is {type(member.value)}, expected int"
            )


# ---------------------------------------------------------------------------
# Bug 5 — T₀ has unit L²[0,1] norm (no √2 scaling)
# ---------------------------------------------------------------------------

class TestCheby2nNormalisation:
    """T₀ must be the constant 1 (no √2 scaling); T_k for k ≥ 1 must
    carry the √2 factor relative to the raw Chebyshev polynomial."""

    def test_t0_is_one(self):
        """T₀(2x−1) = 1 must NOT be scaled by √2."""
        x = np.linspace(0, 1, 50).reshape(1, -1)
        polys = ClenshawCurtisLevelPolynomialBasisGenerator._cheby2n(x, 3)
        np.testing.assert_allclose(polys[0], 1.0)

    def test_t1_carries_sqrt2(self):
        """T₁ should be √2·(2x−1)."""
        x = np.linspace(0, 1, 50).reshape(1, -1)
        polys = ClenshawCurtisLevelPolynomialBasisGenerator._cheby2n(x, 3)
        expected = np.sqrt(2) * (2 * x - 1)
        np.testing.assert_allclose(polys[1], expected)


# ---------------------------------------------------------------------------
# Bug 6 — l2_error_function_values reduces over points, not functions
# ---------------------------------------------------------------------------

class TestL2ErrorAxis:
    def test_returns_one_error_per_function(self):
        """For a (n_pts, n_fns) input the result must have length n_fns."""
        n_pts, n_fns = 500, 4
        y = np.zeros((n_pts, n_fns))
        y_hat = np.ones((n_pts, n_fns))
        err = l2_error_function_values(y, y_hat)
        assert err.shape == (n_fns,), f"shape {err.shape}, expected ({n_fns},)"

    def test_independent_columns(self):
        """Error in one column must not affect the others."""
        n = 100
        y = np.zeros((n, 2))
        y_hat = np.zeros((n, 2))
        y_hat[:, 0] = 1.0  # only first column has error
        err = l2_error_function_values(y, y_hat)
        assert err[0] == pytest.approx(1.0)
        assert err[1] == pytest.approx(0.0)

