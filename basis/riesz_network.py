"""
Conversion of Riesz basis coefficients to ReLU neural networks.

Implements Lemma 5 from Schneider, Ullrich, Vybíral (2025),
"Nonlocal techniques for the analysis of deep ReLU neural network
approximations" (arXiv:2504.04847).

Given a function decomposed in the Riesz basis R_d:

    f(x) = α₀ + Σ_{k ∈ I} [α_k · C_k(x) + β_k · S_k(x)],   x ∈ [0,1]^d,

``coefficients_to_network`` constructs a feed-forward ReLU ANN N with
N(x) = f(x) for all x ∈ [0,1]^d (Lemma 5, wide/shallow variant).

Each C_k, S_k is decomposed into ReLU units via its piecewise-linear
breakpoint structure, then all groups are stacked side-by-side into a
single hidden layer.  The resulting network has depth L = 1 and
width W = Σ_k (2‖k‖₁ + 2) for both C_k and S_k contributions.
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
