"""
Tests for backward direction: ReLU network → Riesz basis coefficients.

Checks:
  - Möbius function and odd-divisor helpers.
  - Breakpoint extraction reproduces known CPL structure.
  - Fourier coefficients match analytic series of C(x) and S(x).
  - Möbius inversion (Fourier → Riesz) recovers known single-basis cases.
  - Full roundtrip: coefficients → network → coefficients.

Visualisation tests (tagged ``visual``) save PNGs to tests/figures/riesz/.
Run with:  pytest tests/test_network_to_coefficients.py -m visual -v -s
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from basis.riesz import eval_C, eval_S
from basis.riesz_network import (
    _mobius,
    _odd_divisors,
    _extract_breakpoints_depth1,
    _fourier_coefficients_cpl,
    network_to_fourier,
    fourier_to_riesz,
    network_to_coefficients,
    coefficients_to_network,
)

# ── output directory for figures ──────────────────────────────────────────────
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "riesz")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Möbius function
# ---------------------------------------------------------------------------

class TestMobius:
    @pytest.mark.parametrize("n, expected", [
        (1, 1), (2, -1), (3, -1), (4, 0), (5, -1), (6, 1),
        (7, -1), (8, 0), (9, 0), (10, 1), (11, -1), (12, 0),
        (30, -1), (105, -1),
    ])
    def test_known_values(self, n, expected):
        assert _mobius(n) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            _mobius(0)


# ---------------------------------------------------------------------------
# Odd divisors
# ---------------------------------------------------------------------------

class TestOddDivisors:
    @pytest.mark.parametrize("n, expected", [
        (1, [1]),
        (2, [1]),
        (3, [1, 3]),
        (4, [1]),
        (6, [1, 3]),
        (12, [1, 3]),
        (15, [1, 3, 5, 15]),
        (24, [1, 3]),
        (45, [1, 3, 5, 9, 15, 45]),
    ])
    def test_known_values(self, n, expected):
        assert _odd_divisors(n) == expected


# ---------------------------------------------------------------------------
# Breakpoint extraction
# ---------------------------------------------------------------------------

class TestBreakpointExtraction:
    def test_c1_breakpoints(self):
        """C_1(x) on [0,1] has one interior breakpoint at 0.5."""
        net = coefficients_to_network(0.0, {(1,): (1.0, 0.0)}, dim=1)
        bps, slopes, intercepts = _extract_breakpoints_depth1(net)
        np.testing.assert_allclose(bps, [0.0, 0.5, 1.0], atol=1e-14)
        np.testing.assert_allclose(slopes, [-4.0, 4.0], atol=1e-10)
        np.testing.assert_allclose(intercepts, [1.0, -3.0], atol=1e-10)

    def test_s1_breakpoints(self):
        """S_1(x) on [0,1] has breakpoints at 0.25 and 0.75."""
        net = coefficients_to_network(0.0, {(1,): (0.0, 1.0)}, dim=1)
        bps, slopes, intercepts = _extract_breakpoints_depth1(net)
        np.testing.assert_allclose(bps, [0.0, 0.25, 0.75, 1.0], atol=1e-14)
        np.testing.assert_allclose(slopes, [4.0, -4.0, 4.0], atol=1e-10)

    def test_reconstruction_matches_network(self):
        """CPL reconstruction from breakpoints matches network evaluation."""
        coeffs = {(1,): (0.5, -0.3), (2,): (0.7, 0.2)}
        net = coefficients_to_network(1.5, coeffs, dim=1)
        bps, slopes, intercepts = _extract_breakpoints_depth1(net)

        x = np.linspace(0, 1, 500).reshape(-1, 1)
        y_net = net(x)

        # Reconstruct from CPL pieces
        y_cpl = np.zeros(500)
        for i in range(len(slopes)):
            mask = (x[:, 0] >= bps[i]) & (x[:, 0] <= bps[i + 1])
            y_cpl[mask] = slopes[i] * x[mask, 0] + intercepts[i]

        np.testing.assert_allclose(y_cpl, y_net, atol=1e-12)


# ---------------------------------------------------------------------------
# Fourier coefficients
# ---------------------------------------------------------------------------

class TestFourierCoefficients:
    def test_c1_fourier(self):
        """C(x) has a_n = 8/(π²n²) for odd n, 0 for even n."""
        net = coefficients_to_network(0.0, {(1,): (1.0, 0.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=20)

        # a_0 should be 0 (C is zero-mean)
        assert abs(a[0]) < 1e-14

        for n in range(1, 21):
            if n % 2 == 1:
                expected = 8.0 / (np.pi**2 * n**2)
                np.testing.assert_allclose(a[n], expected, atol=1e-14,
                                           err_msg=f"a_{n} wrong")
            else:
                np.testing.assert_allclose(a[n], 0.0, atol=1e-14,
                                           err_msg=f"a_{n} should be 0")

        # All sine coefficients should be zero (C is even)
        np.testing.assert_allclose(b, 0.0, atol=1e-14)

    def test_s1_fourier(self):
        """S(x) has b_n = 8·(-1)^{(n-1)/2}/(π²n²) for odd n, 0 for even n."""
        net = coefficients_to_network(0.0, {(1,): (0.0, 1.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=20)

        # a_0 should be 0
        assert abs(a[0]) < 1e-14
        # All cosine coefficients should be zero (S is odd about 1/2)
        np.testing.assert_allclose(a[1:], 0.0, atol=1e-14)

        for n in range(1, 21):
            if n % 2 == 1:
                expected = 8.0 * ((-1) ** ((n - 1) // 2)) / (np.pi**2 * n**2)
                np.testing.assert_allclose(b[n - 1], expected, atol=1e-14,
                                           err_msg=f"b_{n} wrong")
            else:
                np.testing.assert_allclose(b[n - 1], 0.0, atol=1e-14,
                                           err_msg=f"b_{n} should be 0")

    def test_constant_function(self):
        """Constant function has a_0 = value, all else zero."""
        net = coefficients_to_network(3.14, {}, dim=1)
        a, b = network_to_fourier(net, max_freq=10)
        np.testing.assert_allclose(a[0], 3.14, atol=1e-12)
        np.testing.assert_allclose(a[1:], 0.0, atol=1e-14)
        np.testing.assert_allclose(b, 0.0, atol=1e-14)

    def test_c2_fourier(self):
        """C_2(x) = C(2x) has non-zero Fourier coefficients at even freqs."""
        net = coefficients_to_network(0.0, {(2,): (1.0, 0.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=20)

        assert abs(a[0]) < 1e-14
        for n in range(1, 21):
            if n % 2 == 0 and (n // 2) % 2 == 1:
                # n = 2m with m odd: a_n = 8/(π²m²)
                m = n // 2
                expected = 8.0 / (np.pi**2 * m**2)
                np.testing.assert_allclose(a[n], expected, atol=1e-14,
                                           err_msg=f"a_{n} wrong")
            else:
                np.testing.assert_allclose(a[n], 0.0, atol=1e-14,
                                           err_msg=f"a_{n} should be 0")


# ---------------------------------------------------------------------------
# Fourier → Riesz (Möbius inversion)
# ---------------------------------------------------------------------------

class TestFourierToRiesz:
    def test_c1_roundtrip(self):
        """Fourier series of C_1 → Riesz gives α_1 = 1, rest ≈ 0."""
        net = coefficients_to_network(0.0, {(1,): (1.0, 0.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=30)
        alpha_0, coeffs = fourier_to_riesz(a, b, max_k=10)

        assert abs(alpha_0) < 1e-14
        np.testing.assert_allclose(coeffs[(1,)][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(coeffs[(1,)][1], 0.0, atol=1e-10)

        for k in range(2, 11):
            np.testing.assert_allclose(coeffs[(k,)][0], 0.0, atol=1e-10,
                                       err_msg=f"α_{k} should be 0")
            np.testing.assert_allclose(coeffs[(k,)][1], 0.0, atol=1e-10,
                                       err_msg=f"β_{k} should be 0")

    def test_s1_roundtrip(self):
        """Fourier series of S_1 → Riesz gives β_1 = 1, rest ≈ 0."""
        net = coefficients_to_network(0.0, {(1,): (0.0, 1.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=30)
        alpha_0, coeffs = fourier_to_riesz(a, b, max_k=10)

        assert abs(alpha_0) < 1e-14
        np.testing.assert_allclose(coeffs[(1,)][0], 0.0, atol=1e-10)
        np.testing.assert_allclose(coeffs[(1,)][1], 1.0, atol=1e-10)

    def test_c3_roundtrip(self):
        """Fourier series of C_3 → Riesz gives α_3 = 1, rest ≈ 0."""
        net = coefficients_to_network(0.0, {(3,): (1.0, 0.0)}, dim=1)
        a, b = network_to_fourier(net, max_freq=50)
        alpha_0, coeffs = fourier_to_riesz(a, b, max_k=10)

        assert abs(alpha_0) < 1e-14
        for k in range(1, 11):
            expected_a = 1.0 if k == 3 else 0.0
            np.testing.assert_allclose(coeffs[(k,)][0], expected_a, atol=1e-10,
                                       err_msg=f"α_{k} wrong")
            np.testing.assert_allclose(coeffs[(k,)][1], 0.0, atol=1e-10,
                                       err_msg=f"β_{k} should be 0")


# ---------------------------------------------------------------------------
# Full roundtrip: coefficients → network → coefficients
# ---------------------------------------------------------------------------

class TestNetworkToCoefficientsRoundtrip:
    @pytest.mark.parametrize("alpha_0, coeffs, max_freq", [
        (0.0, {(1,): (1.0, 0.0)}, 20),
        (0.0, {(1,): (0.0, 1.0)}, 20),
        (2.0, {(1,): (0.5, -0.3)}, 20),
        (1.5, {(1,): (0.5, -0.3), (2,): (0.7, 0.2)}, 30),
        (0.0, {(1,): (1.0, 1.0), (2,): (-0.5, 0.3), (3,): (0.2, -0.8)}, 50),
    ])
    def test_roundtrip(self, alpha_0, coeffs, max_freq):
        """coefficients_to_network → network_to_coefficients recovers inputs."""
        net = coefficients_to_network(alpha_0, coeffs, dim=1)
        rec_a0, rec_coeffs = network_to_coefficients(net, max_freq)

        np.testing.assert_allclose(rec_a0, alpha_0, atol=1e-10,
                                   err_msg="α₀ mismatch")

        max_k = max(k[0] for k in coeffs)
        for k_tuple, (a_orig, b_orig) in coeffs.items():
            a_rec, b_rec = rec_coeffs[k_tuple]
            np.testing.assert_allclose(a_rec, a_orig, atol=1e-8,
                                       err_msg=f"α_{k_tuple} mismatch")
            np.testing.assert_allclose(b_rec, b_orig, atol=1e-8,
                                       err_msg=f"β_{k_tuple} mismatch")

        # Indices beyond max_k in original should be ≈ 0
        for k in range(max_k + 1, max_freq + 1):
            a_rec, b_rec = rec_coeffs[(k,)]
            np.testing.assert_allclose(a_rec, 0.0, atol=1e-8,
                                       err_msg=f"α_{k} should be 0")
            np.testing.assert_allclose(b_rec, 0.0, atol=1e-8,
                                       err_msg=f"β_{k} should be 0")

    def test_higher_frequency_roundtrip(self):
        """Roundtrip with max_freq=5 basis functions."""
        coeffs = {(k,): (float((-1)**k) / k, float(k % 3) / k)
                  for k in range(1, 6)}
        net = coefficients_to_network(0.5, coeffs, dim=1)
        rec_a0, rec_coeffs = network_to_coefficients(net, max_freq=60)

        np.testing.assert_allclose(rec_a0, 0.5, atol=1e-10)
        for k in range(1, 6):
            a_orig, b_orig = coeffs[(k,)]
            a_rec, b_rec = rec_coeffs[(k,)]
            np.testing.assert_allclose(a_rec, a_orig, atol=1e-8,
                                       err_msg=f"α_{k} mismatch")
            np.testing.assert_allclose(b_rec, b_orig, atol=1e-8,
                                       err_msg=f"β_{k} mismatch")

    def test_function_values_match(self):
        """Recovered coefficients reproduce the same function values."""
        coeffs = {(1,): (0.5, -0.3), (2,): (0.7, 0.2), (3,): (-0.4, 0.6)}
        net = coefficients_to_network(1.5, coeffs, dim=1)
        rec_a0, rec_coeffs = network_to_coefficients(net, max_freq=50)

        # Rebuild network from recovered coefficients (truncate to max_k=3)
        rec_coeffs_trunc = {k: v for k, v in rec_coeffs.items() if k[0] <= 3}
        net_rec = coefficients_to_network(rec_a0, rec_coeffs_trunc, dim=1)

        x = np.linspace(0, 1, 500).reshape(-1, 1)
        y_orig = net(x)
        y_rec = net_rec(x)
        np.testing.assert_allclose(y_rec, y_orig, atol=1e-8)


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------

@pytest.mark.visual
class TestVisualisation:
    def test_plot_roundtrip_coefficients(self):
        """Bar chart: original vs recovered coefficients."""
        coeffs = {(k,): (float((-1)**k) / k, 1.0 / (k + 1))
                  for k in range(1, 9)}
        net = coefficients_to_network(0.5, coeffs, dim=1)
        rec_a0, rec_coeffs = network_to_coefficients(net, max_freq=60)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ks = list(range(1, 9))

        # α_k comparison
        ax = axes[0]
        a_orig = [coeffs[(k,)][0] for k in ks]
        a_rec = [rec_coeffs[(k,)][0] for k in ks]
        x_pos = np.arange(len(ks))
        ax.bar(x_pos - 0.15, a_orig, 0.3, label="Original", color="steelblue")
        ax.bar(x_pos + 0.15, a_rec, 0.3, label="Recovered", color="coral")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$k$")
        ax.set_ylabel("$\\alpha_k$")
        ax.set_title("Cosine coefficients")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # β_k comparison
        ax = axes[1]
        b_orig = [coeffs[(k,)][1] for k in ks]
        b_rec = [rec_coeffs[(k,)][1] for k in ks]
        ax.bar(x_pos - 0.15, b_orig, 0.3, label="Original", color="steelblue")
        ax.bar(x_pos + 0.15, b_rec, 0.3, label="Recovered", color="coral")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$k$")
        ax.set_ylabel("$\\beta_k$")
        ax.set_title("Sine coefficients")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Function reconstruction
        ax = axes[2]
        x_plot = np.linspace(0, 1, 500).reshape(-1, 1)
        y_orig = net(x_plot)
        rec_trunc = {k: v for k, v in rec_coeffs.items() if k[0] <= 8}
        net_rec = coefficients_to_network(rec_a0, rec_trunc, dim=1)
        y_rec = net_rec(x_plot)
        ax.plot(x_plot, y_orig, "b-", lw=1.5, label="Original network")
        ax.plot(x_plot, y_rec, "r--", lw=1.5, label="Reconstructed")
        ax.set_xlabel("$x$")
        ax.set_title("Function comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Network → Riesz coefficients roundtrip", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG_DIR, "roundtrip_coefficients.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
