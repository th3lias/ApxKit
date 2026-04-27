"""
Conversion between Riesz basis coefficients and ReLU neural networks.

Implements Lemma 5 and Equations (8)–(9) from Schneider, Ullrich, Vybíral
(2025), "Nonlocal techniques for the analysis of deep ReLU neural network
approximations" (arXiv:2504.04847).

**Forward direction** (Lemma 5):
    ``coefficients_to_network`` constructs a feed-forward ReLU ANN N with
    N(x) = f(x) for all x ∈ [0,1]^d.

**Backward direction** (Eqs. 8–9):
    ``network_to_coefficients`` recovers Riesz basis coefficients from a
    depth-1, d=1 ReLU network via:
    Step 1: closed-form Fourier integration of the piecewise-linear function,
    Step 2: Möbius inversion (Fourier → Riesz).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# ReLU network container
# ---------------------------------------------------------------------------

class ReLUNetwork:
    """
    Feed-forward ReLU neural network: A^(L) ∘ ReLU ∘ … ∘ ReLU ∘ A^(0).

    Attributes
    ----------
    weights : list[np.ndarray]
        Weight matrices.  ``weights[i]`` has shape ``(n_{i+1}, n_i)``.
    biases : list[np.ndarray]
        Bias vectors.  ``biases[i]`` has shape ``(n_{i+1},)``.
    depth : int
        Number of hidden layers (``len(weights) − 1``).
    width : int
        Maximum hidden-layer width.
    """

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]):
        assert len(weights) == len(biases) and len(weights) >= 2
        self.weights = weights
        self.biases = biases
        self.depth = len(weights) - 1
        self.width = max(w.shape[0] for w in weights[:-1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the network on a batch of inputs.

        Parameters
        ----------
        x : (N, d) or (d,)

        Returns
        -------
        (N,) or scalar
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        h = x
        for i in range(len(self.weights) - 1):
            h = h @ self.weights[i].T + self.biases[i]
            np.maximum(h, 0.0, out=h)
        h = h @ self.weights[-1].T + self.biases[-1]

        out = h.squeeze(-1)
        return out[0] if single else out

    def to_torch(self):
        """Convert to ``torch.nn.Sequential`` (requires PyTorch)."""
        import torch
        import torch.nn as nn

        layers: list[nn.Module] = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            linear = nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(W.astype(np.float64)))
                linear.bias.copy_(torch.from_numpy(b.astype(np.float64)))
            layers.append(linear)
            if i < len(self.weights) - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __repr__(self) -> str:
        d_in = self.weights[0].shape[1]
        n_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        return (f"ReLUNetwork(d={d_in}, width={self.width}, "
                f"depth={self.depth}, params={n_params})")


# ---------------------------------------------------------------------------
# Per-basis-function neuron construction
# ---------------------------------------------------------------------------

def _c_neurons(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Single-hidden-layer ReLU representation of C_k(x) = C(k·x).

    C(t) has slope ±4 with breakpoints at every multiple of 1/2.

    Returns
    -------
    W_h : (n_hidden, d)
    b_h : (n_hidden,)
    w_o : (n_hidden,)
    b_o : float
    """
    k = np.asarray(k, dtype=float)
    d = k.size

    t_min = float(sum(min(0.0, ki) for ki in k))
    t_max = float(sum(max(0.0, ki) for ki in k))

    # Interior breakpoints of C(t): at n/2 for integers n
    n_lo = math.ceil(2.0 * t_min)
    n_hi = math.floor(2.0 * t_max)
    interior = [(n, n / 2.0) for n in range(n_lo, n_hi + 1)
                if t_min + 1e-12 < n / 2.0 < t_max - 1e-12]

    # Value and slope of C at t_min
    frac = t_min - math.floor(t_min)
    c_val = 4.0 * abs(frac - 0.5) - 1.0
    if abs(frac) < 1e-12:
        slope0 = -4.0
    elif abs(frac - 0.5) < 1e-12:
        slope0 = 4.0
    elif frac < 0.5:
        slope0 = -4.0
    else:
        slope0 = 4.0

    bp_vals = [bp for _, bp in interior]
    deltas = [-8.0 if n % 2 == 0 else 8.0 for n, _ in interior]

    n_bp = len(bp_vals)
    n_hidden = n_bp + 2

    W_h = np.zeros((n_hidden, d))
    b_h = np.zeros(n_hidden)
    W_h[0] = k;   b_h[0] = 0.0
    W_h[1] = -k;  b_h[1] = 0.0
    for j, bp in enumerate(bp_vals):
        W_h[2 + j] = k
        b_h[2 + j] = -bp

    w_o = np.zeros(n_hidden)
    w_o[0] = slope0
    w_o[1] = -slope0
    w_o[2:2 + n_bp] = deltas
    b_o = c_val - slope0 * t_min

    return W_h, b_h, w_o, b_o


def _s_neurons(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Single-hidden-layer ReLU representation of S_k(x) = S(k·x).

    S(t) has slope ±4 with breakpoints at 1/4 + n/2 for every integer n.

    Returns
    -------
    W_h : (n_hidden, d)
    b_h : (n_hidden,)
    w_o : (n_hidden,)
    b_o : float
    """
    k = np.asarray(k, dtype=float)
    d = k.size

    t_min = float(sum(min(0.0, ki) for ki in k))
    t_max = float(sum(max(0.0, ki) for ki in k))

    n_lo = math.ceil(2.0 * t_min - 0.5)
    n_hi = math.floor(2.0 * t_max - 0.5)
    interior = [(n, 0.25 + n / 2.0) for n in range(n_lo, n_hi + 1)
                if t_min + 1e-12 < 0.25 + n / 2.0 < t_max - 1e-12]

    frac = t_min - math.floor(t_min)
    s_val = abs(2.0 - 4.0 * abs(frac - 0.25)) - 1.0
    if abs(frac - 0.25) < 1e-12:
        slope0 = -4.0
    elif abs(frac - 0.75) < 1e-12:
        slope0 = 4.0
    elif frac < 0.25:
        slope0 = 4.0
    elif frac < 0.75:
        slope0 = -4.0
    else:
        slope0 = 4.0

    bp_vals = [bp for _, bp in interior]
    deltas = [-8.0 if n % 2 == 0 else 8.0 for n, _ in interior]

    n_bp = len(bp_vals)
    n_hidden = n_bp + 2

    W_h = np.zeros((n_hidden, d))
    b_h = np.zeros(n_hidden)
    W_h[0] = k;   b_h[0] = 0.0
    W_h[1] = -k;  b_h[1] = 0.0
    for j, bp in enumerate(bp_vals):
        W_h[2 + j] = k
        b_h[2 + j] = -bp

    w_o = np.zeros(n_hidden)
    w_o[0] = slope0
    w_o[1] = -slope0
    w_o[2:2 + n_bp] = deltas
    b_o = s_val - slope0 * t_min

    return W_h, b_h, w_o, b_o


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def coefficients_to_network(
    alpha_0: float,
    coefficients: dict[tuple, tuple[float, float]],
    dim: int,
) -> ReLUNetwork:
    """
    Convert Riesz basis coefficients to a ReLU neural network (Lemma 5).

    Constructs an ANN N such that for all x ∈ [0,1]^d:

        N(x) = α₀ + Σ_{k ∈ I} [α_k · C_k(x) + β_k · S_k(x)].

    Parameters
    ----------
    alpha_0 : float
        Coefficient of the constant basis function.
    coefficients : dict
        Maps each k (tuple of ints) to (α_k, β_k).
    dim : int
        Input dimension d.

    Returns
    -------
    ReLUNetwork
        Single-hidden-layer ReLU network with exact reproduction.

    Notes
    -----
    Uses the *wide* (shallow) construction of Lemma 5: each C_k, S_k is
    reproduced by a sub-network and all sub-networks are stacked
    side-by-side into a single hidden layer.
    """
    if not coefficients:
        W0 = np.zeros((1, dim))
        b0 = np.zeros(1)
        W1 = np.zeros((1, 1))
        b1 = np.array([alpha_0])
        return ReLUNetwork([W0, W1], [b0, b1])

    hidden_W_rows: list[np.ndarray] = []
    hidden_b_parts: list[np.ndarray] = []
    output_w_parts: list[np.ndarray] = []
    output_bias = alpha_0

    for k_tuple, (alpha_k, beta_k) in coefficients.items():
        k = np.array(k_tuple, dtype=float)
        assert k.size == dim, f"Index k={k_tuple} has {k.size} entries but dim={dim}"

        if abs(alpha_k) > 0.0:
            W_h, b_h, w_o, b_o = _c_neurons(k)
            hidden_W_rows.append(W_h)
            hidden_b_parts.append(b_h)
            output_w_parts.append(alpha_k * w_o)
            output_bias += alpha_k * b_o

        if abs(beta_k) > 0.0:
            W_h, b_h, w_o, b_o = _s_neurons(k)
            hidden_W_rows.append(W_h)
            hidden_b_parts.append(b_h)
            output_w_parts.append(beta_k * w_o)
            output_bias += beta_k * b_o

    W_hidden = np.vstack(hidden_W_rows)
    b_hidden = np.concatenate(hidden_b_parts)
    w_output = np.concatenate(output_w_parts)
    W_output = w_output.reshape(1, -1)
    b_output = np.array([output_bias])

    return ReLUNetwork([W_hidden, W_output], [b_hidden, b_output])


# ---------------------------------------------------------------------------
# Backward direction: Network → Riesz coefficients (Eqs. 8–9)
# ---------------------------------------------------------------------------

def _mobius(n: int) -> int:
    """Möbius function μ(n).

    Returns 1 if *n* is a product of an even number of distinct primes,
    −1 for an odd number, and 0 if any prime factor is squared.
    """
    if n <= 0:
        raise ValueError(f"Möbius function requires n > 0, got {n}")
    if n == 1:
        return 1
    count = 0
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            temp //= d
            if temp % d == 0:
                return 0
            count += 1
        d += 1
    if temp > 1:
        count += 1
    return 1 if count % 2 == 0 else -1


def _odd_divisors(n: int) -> list[int]:
    """Return all odd divisors of *n* in ascending order."""
    # Strip factors of 2
    while n % 2 == 0:
        n //= 2
    # n is now odd; collect all divisors
    divs: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
        d += 1
    return sorted(divs)


def _extract_breakpoints_depth1(
    net: ReLUNetwork,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract piecewise-linear structure of a depth-1, d=1 network on [0,1].

    Returns
    -------
    breakpoints : (n_pieces + 1,)
        Sorted boundary points including 0.0 and 1.0.
    slopes : (n_pieces,)
    intercepts : (n_pieces,)
        On each piece, f(x) = slope·x + intercept.
    """
    if net.depth != 1:
        raise ValueError(f"Expected depth-1 network, got depth {net.depth}")
    d_in = net.weights[0].shape[1]
    if d_in != 1:
        raise ValueError(f"Expected d=1 network, got d={d_in}")

    w_h = net.weights[0][:, 0]          # (n_hidden,)
    b_h = net.biases[0]                  # (n_hidden,)
    w_out = net.weights[-1][0]           # (n_hidden,)
    b_out = float(net.biases[-1][0])

    # Collect kink locations
    bps_set: set[float] = set()
    for j in range(len(w_h)):
        if abs(w_h[j]) > 1e-15:
            bp = -b_h[j] / w_h[j]
            if 1e-15 < bp < 1.0 - 1e-15:
                bps_set.add(float(bp))

    # Sort and deduplicate within tolerance
    bps_sorted = sorted([0.0] + list(bps_set) + [1.0])
    deduped = [bps_sorted[0]]
    for bp in bps_sorted[1:]:
        if bp - deduped[-1] > 1e-14:
            deduped.append(bp)
    bps = np.array(deduped)

    n_pieces = len(bps) - 1
    slopes = np.zeros(n_pieces)
    intercepts = np.zeros(n_pieces)

    for i in range(n_pieces):
        x_mid = (bps[i] + bps[i + 1]) / 2.0
        active = (w_h * x_mid + b_h) > 0
        slopes[i] = float(np.dot(w_out[active], w_h[active]))
        # Exact value at left endpoint
        h_left = np.maximum(w_h * bps[i] + b_h, 0.0)
        y_left = float(np.dot(w_out, h_left) + b_out)
        intercepts[i] = y_left - slopes[i] * bps[i]

    return bps, slopes, intercepts


def _fourier_coefficients_cpl(
    breakpoints: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Fourier coefficients of a piecewise-linear function on [0,1].

    Parameters
    ----------
    breakpoints : (n_pieces + 1,)
    slopes, intercepts : (n_pieces,)
        f(x) = slopes[i]·x + intercepts[i] on [breakpoints[i], breakpoints[i+1]].
    max_freq : int
        Compute a_0 … a_{max_freq} and b_1 … b_{max_freq}.

    Returns
    -------
    a : (max_freq + 1,)
        Cosine coefficients.  a[0] = ∫₀¹ f(x) dx.
    b : (max_freq,)
        Sine coefficients.  b[n−1] corresponds to frequency n.
    """
    a = np.zeros(max_freq + 1)
    b = np.zeros(max_freq)

    x0 = breakpoints[:-1]
    x1 = breakpoints[1:]

    # a_0 = ∫₀¹ f(x) dx
    a[0] = float(np.sum(slopes / 2.0 * (x1**2 - x0**2)
                        + intercepts * (x1 - x0)))

    # a_n, b_n for n ≥ 1 via closed-form antiderivatives:
    #   ∫(αx+β)cos(ωx)dx = α cos(ωx)/ω² + (αx+β) sin(ωx)/ω
    #   ∫(αx+β)sin(ωx)dx = α sin(ωx)/ω² − (αx+β) cos(ωx)/ω
    f_x0 = slopes * x0 + intercepts
    f_x1 = slopes * x1 + intercepts

    for n in range(1, max_freq + 1):
        omega = 2.0 * np.pi * n
        omega2 = omega * omega

        cos_x0 = np.cos(omega * x0)
        cos_x1 = np.cos(omega * x1)
        sin_x0 = np.sin(omega * x0)
        sin_x1 = np.sin(omega * x1)

        cos_integral = float(np.sum(
            slopes * cos_x1 / omega2 + f_x1 * sin_x1 / omega
            - slopes * cos_x0 / omega2 - f_x0 * sin_x0 / omega
        ))
        sin_integral = float(np.sum(
            slopes * sin_x1 / omega2 - f_x1 * cos_x1 / omega
            - slopes * sin_x0 / omega2 + f_x0 * cos_x0 / omega
        ))

        a[n] = 2.0 * cos_integral
        b[n - 1] = 2.0 * sin_integral

    return a, b


def network_to_fourier(
    net: ReLUNetwork,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Fourier coefficients of a depth-1, d=1 ReLU network on [0,1].

    Parameters
    ----------
    net : ReLUNetwork
        Must have depth = 1 and input dimension d = 1.
    max_freq : int
        Maximum Fourier frequency.

    Returns
    -------
    a : (max_freq + 1,)
        Cosine coefficients.
    b : (max_freq,)
        Sine coefficients.
    """
    bps, slopes, intercepts = _extract_breakpoints_depth1(net)
    return _fourier_coefficients_cpl(bps, slopes, intercepts, max_freq)


def fourier_to_riesz(
    a: np.ndarray,
    b: np.ndarray,
    max_k: int,
) -> tuple[float, dict[tuple[int, ...], tuple[float, float]]]:
    """Convert Fourier coefficients to Riesz basis coefficients (Eqs. 8–9).

    Uses the Möbius inversion:

        α_k = (π²/8) Σ_{odd d | k} μ(d)/d² · a_{k/d}
        β_k = (π²/8) Σ_{odd d | k} (−1)^{(d−1)/2} μ(d)/d² · b_{k/d}

    Parameters
    ----------
    a : (max_freq + 1,)
        Cosine Fourier coefficients.
    b : (max_freq,)
        Sine Fourier coefficients (b[n−1] ↔ frequency n).
    max_k : int
        Maximum Riesz index.

    Returns
    -------
    alpha_0 : float
    coefficients : dict mapping ``(k,)`` → ``(α_k, β_k)``
    """
    alpha_0 = float(a[0])
    norm = np.pi ** 2 / 8.0
    coefficients: dict[tuple[int, ...], tuple[float, float]] = {}

    for k in range(1, max_k + 1):
        alpha_k = 0.0
        beta_k = 0.0
        for d in _odd_divisors(k):
            mu_d = _mobius(d)
            if mu_d == 0:
                continue
            m = k // d
            weight = mu_d / (d * d)
            if m < len(a):
                alpha_k += weight * a[m]
            if 1 <= m <= len(b):
                beta_k += ((-1) ** ((d - 1) // 2)) * weight * b[m - 1]
        coefficients[(k,)] = (norm * alpha_k, norm * beta_k)

    return alpha_0, coefficients


def network_to_coefficients(
    net: ReLUNetwork,
    max_freq: int,
) -> tuple[float, dict[tuple[int, ...], tuple[float, float]]]:
    """Convert a depth-1, d=1 ReLU network to Riesz basis coefficients.

    Pipeline: Network → Fourier coefficients → Riesz coefficients.

    Parameters
    ----------
    net : ReLUNetwork
        Must have depth = 1 and input dimension d = 1.
    max_freq : int
        Maximum frequency / Riesz index.

    Returns
    -------
    alpha_0 : float
    coefficients : dict mapping ``(k,)`` → ``(α_k, β_k)``

    See Also
    --------
    coefficients_to_network : the forward direction.
    """
    a, b = network_to_fourier(net, max_freq)
    return fourier_to_riesz(a, b, max_freq)
