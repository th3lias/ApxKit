"""
Tests for Riesz basis → ReLU network conversion (Lemma 5).

Checks:
  - Individual neuron groups (_c_neurons, _s_neurons) reproduce C_k, S_k.
  - coefficients_to_network reproduces linear combinations exactly.
  - Multivariate (2-D, 3-D) and negative-index cases.
  - Network properties (depth, width) match Lemma 5 predictions.

Visualisation tests (tagged ``visual``) save PNGs to tests/figures/riesz/.
Run with:  pytest tests/test_riesz_network.py -m visual -v -s
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from basis.riesz import eval_C, eval_S
from basis.riesz_network import (
    coefficients_to_network, ReLUNetwork, _c_neurons, _s_neurons,
)

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "riesz")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-basis neuron tests: C_k
# ---------------------------------------------------------------------------

class TestCNeurons:
    @pytest.mark.parametrize("k_tuple", [(1,), (2,), (3,), (5,)])
    def test_c_k_univariate(self, k_tuple):
        k = np.array(k_tuple, dtype=float)
        W_h, b_h, w_o, b_o = _c_neurons(k)
        x = np.linspace(0, 1, 200).reshape(-1, 1)
        hidden = np.maximum(x @ W_h.T + b_h, 0.0)
        network_vals = hidden @ w_o + b_o
        ref_vals = eval_C(x[:, 0] * k[0])
        np.testing.assert_allclose(network_vals, ref_vals, atol=1e-10,
                                   err_msg=f"C_k failed for k={k_tuple}")

    def test_c_k_multivariate(self):
        k = np.array([1, -1], dtype=float)
        W_h, b_h, w_o, b_o = _c_neurons(k)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=(500, 2))
        hidden = np.maximum(x @ W_h.T + b_h, 0.0)
        network_vals = hidden @ w_o + b_o
        np.testing.assert_allclose(network_vals, eval_C(x @ k), atol=1e-10)

    def test_c_k_multivariate_positive(self):
        k = np.array([2, 3], dtype=float)
        W_h, b_h, w_o, b_o = _c_neurons(k)
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 1, size=(500, 2))
        hidden = np.maximum(x @ W_h.T + b_h, 0.0)
        np.testing.assert_allclose(hidden @ w_o + b_o, eval_C(x @ k), atol=1e-10)


# ---------------------------------------------------------------------------
# Single-basis neuron tests: S_k
# ---------------------------------------------------------------------------

class TestSNeurons:
    @pytest.mark.parametrize("k_tuple", [(1,), (2,), (3,), (5,)])
    def test_s_k_univariate(self, k_tuple):
        k = np.array(k_tuple, dtype=float)
        W_h, b_h, w_o, b_o = _s_neurons(k)
        x = np.linspace(0, 1, 200).reshape(-1, 1)
        hidden = np.maximum(x @ W_h.T + b_h, 0.0)
        network_vals = hidden @ w_o + b_o
        np.testing.assert_allclose(network_vals, eval_S(x[:, 0] * k[0]), atol=1e-10,
                                   err_msg=f"S_k failed for k={k_tuple}")

    def test_s_k_multivariate(self):
        k = np.array([1, -1], dtype=float)
        W_h, b_h, w_o, b_o = _s_neurons(k)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=(500, 2))
        hidden = np.maximum(x @ W_h.T + b_h, 0.0)
        np.testing.assert_allclose(hidden @ w_o + b_o, eval_S(x @ k), atol=1e-10)


# ---------------------------------------------------------------------------
# Full network conversion tests
# ---------------------------------------------------------------------------

class TestCoefficientsToNetwork:
    def test_constant_function(self):
        net = coefficients_to_network(3.14, {}, dim=1)
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        np.testing.assert_allclose(net(x), 3.14, atol=1e-12)

    def test_single_C1(self):
        net = coefficients_to_network(0.0, {(1,): (1.0, 0.0)}, dim=1)
        x = np.linspace(0, 1, 200).reshape(-1, 1)
        np.testing.assert_allclose(net(x), eval_C(x[:, 0]), atol=1e-10)

    def test_single_S1(self):
        net = coefficients_to_network(0.0, {(1,): (0.0, 1.0)}, dim=1)
        x = np.linspace(0, 1, 200).reshape(-1, 1)
        np.testing.assert_allclose(net(x), eval_S(x[:, 0]), atol=1e-10)

    def test_linear_combination_1d(self):
        """f(x) = 2 + 0.5·C_1(x) − 0.3·S_1(x) + 0.7·C_2(x)."""
        coeffs = {(1,): (0.5, -0.3), (2,): (0.7, 0.0)}
        net = coefficients_to_network(2.0, coeffs, dim=1)
        x = np.linspace(0, 1, 300).reshape(-1, 1)
        ref = (2.0 + 0.5 * eval_C(x[:, 0]) - 0.3 * eval_S(x[:, 0])
               + 0.7 * eval_C(2.0 * x[:, 0]))
        np.testing.assert_allclose(net(x), ref, atol=1e-10)

    def test_multivariate_2d(self):
        """f(x) = 1 + C_{(1,0)} + S_{(0,1)} + 0.5·C_{(1,1)}."""
        coeffs = {(1, 0): (1.0, 0.0), (0, 1): (0.0, 1.0), (1, 1): (0.5, 0.0)}
        net = coefficients_to_network(1.0, coeffs, dim=2)
        rng = np.random.default_rng(99)
        x = rng.uniform(0, 1, size=(500, 2))
        ref = (1.0 + eval_C(x[:, 0]) + eval_S(x[:, 1])
               + 0.5 * eval_C(x[:, 0] + x[:, 1]))
        np.testing.assert_allclose(net(x), ref, atol=1e-10)

    def test_negative_index(self):
        """k = (1, −1) in 2-D."""
        coeffs = {(1, -1): (1.0, 1.0)}
        net = coefficients_to_network(0.0, coeffs, dim=2)
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, size=(500, 2))
        t = x[:, 0] - x[:, 1]
        np.testing.assert_allclose(net(x), eval_C(t) + eval_S(t), atol=1e-10)

    def test_higher_frequency(self):
        """f(x) = C_5(x) + S_5(x), univariate."""
        coeffs = {(5,): (1.0, 1.0)}
        net = coefficients_to_network(0.0, coeffs, dim=1)
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        ref = eval_C(5.0 * x[:, 0]) + eval_S(5.0 * x[:, 0])
        np.testing.assert_allclose(net(x), ref, atol=1e-10)

    def test_single_point_evaluation(self):
        net = coefficients_to_network(1.0, {(1,): (1.0, 0.0)}, dim=1)
        val = net(np.array([0.25]))
        assert val == pytest.approx(1.0 + eval_C(0.25), abs=1e-12)

    def test_3d(self):
        coeffs = {(1, 0, 0): (1.0, 0.0), (0, 1, 0): (0.0, 1.0), (1, 1, 1): (0.5, -0.5)}
        net = coefficients_to_network(0.0, coeffs, dim=3)
        rng = np.random.default_rng(123)
        x = rng.uniform(0, 1, size=(200, 3))
        k111 = np.array([1, 1, 1], dtype=float)
        ref = (eval_C(x[:, 0]) + eval_S(x[:, 1])
               + 0.5 * eval_C(x @ k111) - 0.5 * eval_S(x @ k111))
        np.testing.assert_allclose(net(x), ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Network properties (Lemma 5 bounds)
# ---------------------------------------------------------------------------

class TestNetworkProperties:
    def test_depth_is_one(self):
        coeffs = {(1,): (1.0, 1.0), (2,): (1.0, 1.0), (3,): (1.0, 1.0)}
        net = coefficients_to_network(0.0, coeffs, dim=1)
        assert net.depth == 1

    def test_width_positive(self):
        net = coefficients_to_network(0.0, {(1,): (1.0, 1.0)}, dim=1)
        assert net.width > 0

    def test_repr(self):
        net = coefficients_to_network(0.0, {(1,): (1.0, 0.0)}, dim=1)
        r = repr(net)
        assert "ReLUNetwork" in r
        assert "depth=1" in r


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    def test_plot_network_verification(self):
        """6-panel verification: individual basis functions + linear combination."""
        x = np.linspace(0, 1, 500).reshape(-1, 1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Top row: individual basis functions
        for col, k_val in enumerate([1, 2, 3]):
            ax = axes[0, col]
            ref_c = eval_C(k_val * x[:, 0])
            ref_s = eval_S(k_val * x[:, 0])
            net_c = coefficients_to_network(0.0, {(k_val,): (1.0, 0.0)}, dim=1)
            net_s = coefficients_to_network(0.0, {(k_val,): (0.0, 1.0)}, dim=1)
            ax.plot(x, ref_c, "b-", lw=1.2, label=f"$C_{k_val}$ (direct)")
            ax.plot(x, net_c(x), "b--", lw=1.2, label=f"$C_{k_val}$ (network)")
            ax.plot(x, ref_s, "r-", lw=1.2, label=f"$S_{k_val}$ (direct)")
            ax.plot(x, net_s(x), "r--", lw=1.2, label=f"$S_{k_val}$ (network)")
            ax.set_title(f"k = {k_val}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Bottom left: linear combination
        coeffs = {(1,): (0.5, -0.3), (2,): (0.7, 0.2), (3,): (-0.4, 0.6)}
        net = coefficients_to_network(1.5, coeffs, dim=1)
        ref = 1.5
        for k, (a, b) in coeffs.items():
            ref = ref + a * eval_C(k[0] * x[:, 0]) + b * eval_S(k[0] * x[:, 0])
        axes[1, 0].plot(x, ref, "b-", lw=1.2, label="direct")
        axes[1, 0].plot(x, net(x), "r--", lw=1.2, label="network")
        axes[1, 0].set_title("6-term combination")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom middle: pointwise error
        err = np.abs(ref - net(x))
        axes[1, 1].semilogy(x, np.maximum(err, 1e-17), "purple", lw=1)
        axes[1, 1].axhline(np.finfo(float).eps, color="red", ls=":", alpha=0.6,
                            label="machine $\\varepsilon$")
        axes[1, 1].set_title("Reconstruction error")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(True, alpha=0.3)

        # Bottom right: architecture summary
        ax = axes[1, 2]
        ax.axis("off")
        net_10 = coefficients_to_network(0.0, {(10,): (1.0, 1.0)}, dim=1)
        text = (
            f"6-term combination:\n  {net}\n\n"
            f"$C_{{10}} + S_{{10}}$:\n  {net_10}\n\n"
            f"Neurons per $C_k$: $\\approx 2\\|k\\|_1 + 2$"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        ax.set_title("Architecture summary")

        fig.suptitle("Riesz basis → ReLU network verification", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "network_verification.png"), dpi=150)
        plt.close(fig)
