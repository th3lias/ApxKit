# Riesz Basis ↔ ReLU Network Conversion

## Background: The Riesz Basis System R_d

The paper [Schneider, Ullrich, Vybíral 2025] introduces the multivariate Riesz basis system (Eq. 3)

$$\mathcal{R}_d := \{1\} \cup \{\mathcal{C}_k,\, \mathcal{S}_k : k \in \mathbb{Z}^d,\, k \succ 0\}$$

built from two univariate piecewise-linear functions on [0, 1], extended periodically (Eq. 1):

$$\mathcal{C}(x) := 4\left|x - \tfrac{1}{2}\right| - 1 = \begin{cases} 1 - 4x, & x \in [0, \tfrac{1}{2}],\\ 4x - 3, & x \in [\tfrac{1}{2}, 1], \end{cases}$$

$$\mathcal{S}(x) := \left|2 - 4\left|x - \tfrac{1}{4}\right|\right| - 1 = \begin{cases} 4x, & x \in [0, \tfrac{1}{4}],\\ 2 - 4x, & x \in [\tfrac{1}{4}, \tfrac{3}{4}],\\ 4x - 4, & x \in [\tfrac{3}{4}, 1]. \end{cases}$$

These interpolate cos(2πx) and sin(2πx) at {0, 1/4, 1/2, 3/4, 1}, but are piecewise linear rather than smooth. The multivariate basis functions are defined via inner products:

$$\mathcal{C}_k(x) := \mathcal{C}(k \cdot x), \qquad \mathcal{S}_k(x) := \mathcal{S}(k \cdot x), \qquad k \in \mathbb{Z}^d,\ x \in [0,1]^d.$$

The key result from [53] is that R_d forms a Riesz basis of L_2([0,1]^d) for every d >= 2, and that the Riesz constants c, C in

$$c \sum_{k \succ 0}(\alpha_k^2 + \beta_k^2) \;\le\; \left\|\sum_{k \succ 0}(\alpha_k \mathcal{C}_k + \beta_k \mathcal{S}_k)\right\|_2^2 \;\le\; C \sum_{k \succ 0}(\alpha_k^2 + \beta_k^2)$$

can be chosen independently of d (with c = 1/6, C = 1/2). This extends to Sobolev spaces W^s([0,1]^d) and Barron classes B^s([0,1]^d) for 0 < s < 1 (Theorems 2 and 3).

## Why This Basis is Useful for Neural Networks

The crucial property is that C and S are **piecewise linear** — the same function class that ReLU networks compute. Any composition of affine maps and ReLU activations produces a continuous piecewise-linear function, and conversely, any continuous piecewise-linear function can be represented exactly by a ReLU network.

Concretely, the hat function H (Eq. 21) provides the elementary building block:

$$H(x) = \begin{cases} 2x, & 0 \le x \le \tfrac{1}{2},\\ 2(1-x), & \tfrac{1}{2} < x \le 1, \end{cases} = (2, -4) \cdot \operatorname{ReLU}\!\left(\begin{pmatrix}1\\1\end{pmatrix}x + \begin{pmatrix}0\\-\tfrac{1}{2}\end{pmatrix}\right).$$

Since C = 1 - 2H, the basis function C is immediately a width-2, depth-1 ReLU network. By composing hat functions (H doubles the oscillation frequency each time), the paper constructs C_k, S_k as deep narrow networks with width 2 and depth O(log ||k||_1).

## What `coefficients_to_network` Implements

Given a finite decomposition in the Riesz basis (Eq. 13/22),

$$f(x) = \alpha_0 + \sum_{k \in I}\bigl[\alpha_k\,\mathcal{C}_k(x) + \beta_k\,\mathcal{S}_k(x)\bigr], \qquad x \in [0,1]^d,$$

the function `coefficients_to_network(alpha_0, coefficients, dim)` constructs a feed-forward ReLU ANN N such that N(x) = f(x) for all x in [0, 1]^d. This is the content of **Lemma 5** in the paper.

### Construction details

The implementation uses the **wide (first) architecture** from Lemma 5. Rather than the deep hat-function composition (which gives width W = 4 #I and depth L = 4 + log_2(max ||k||_1)), we use a **shallow variant**: each C_k and S_k is represented by a single hidden layer exploiting the piecewise-linear structure directly.

**Step 1 — ReLU decomposition of C(t) on a bounded interval.**
C(t) is piecewise linear with slope alternating between -4 and +4, with breakpoints (kinks) at every multiple of 1/2. On any interval [t_min, t_max], it can be written as:

$$\mathcal{C}(t) = \mathcal{C}(t_{\min}) + s_0\,(t - t_{\min}) + \sum_{j} \Delta s_j \cdot \operatorname{ReLU}(t - t_j)$$

where t_j are the interior breakpoints and Delta s_j in {-8, +8} are the slope changes. The analogous decomposition holds for S(t) with breakpoints at 1/4 + n/2.

**Step 2 — From scalar to multivariate.**
For C_k(x) = C(k . x) with k in Z^d, the argument t = k . x is a linear function of x in [0,1]^d. The range is [t_min, t_max] where t_min = sum min(0, k_i) and t_max = sum max(0, k_i). Each ReLU(k . x - t_j) becomes a single hidden neuron with weight vector k and bias -t_j. For the linear term k . x, we use the identity k . x = ReLU(k . x) - ReLU(-k . x) (two additional neurons).

This gives **2 ||k||_1 + 2 hidden neurons** per basis function in the univariate case (2 ||k||_1 breakpoints on [0, ||k||_1], plus 2 for the linear term), and similarly for the multivariate case.

**Step 3 — Combining all terms (Lemma 5).**
All per-basis neuron groups are stacked side-by-side into a single hidden layer. Each group is self-contained: it reads the shared input x but has its own weights and biases, and computes one basis function value independently.

To make this concrete, consider an example with I = {k_1, k_2} in 1D. Suppose C_{k_1} needs n_1 hidden neurons and S_{k_1} needs n_2, and so on. The full hidden layer is a block-diagonal structure:

```
                ┌─────────────────────────────────────────────────────────┐
                │  neurons for   │  neurons for   │  neurons for   │ ... │
  x ∈ R^d  ──►  │  C_{k_1}       │  S_{k_1}       │  C_{k_2}       │     │
                │  (n_1 units)   │  (n_2 units)   │  (n_3 units)   │     │
                └──────┬─────────┴──────┬─────────┴──────┬─────────┘
                       │                │                │
                   ReLU(W^1_h x + b^1_h)  ReLU(W^2_h x + b^2_h)  ...
                       │                │                │
                       ▼                ▼                ▼
         group 1 output:       group 2 output:    group 3 output:
         w^1_o · h_1 + b^1_o  w^2_o · h_2 + b^2_o  ...
              = C_{k_1}(x)         = S_{k_1}(x)       = C_{k_2}(x)
```

Each group's output is a scalar that equals the corresponding basis function evaluated at x. The groups do not interact in the hidden layer — they share the input x but have independent weight vectors (all proportional to k) and independent biases (encoding the breakpoints).

The output layer then simply takes the weighted sum over all group outputs:

$$\mathcal{N}(x) = \alpha_0 + \sum_{k \in I}\Bigl[\alpha_k \cdot \underbrace{\bigl(w^{\mathcal{C}_k}_o \cdot \operatorname{ReLU}(W^{\mathcal{C}_k}_h\, x + b^{\mathcal{C}_k}_h) + b^{\mathcal{C}_k}_o\bigr)}_{= \mathcal{C}_k(x)} + \beta_k \cdot \underbrace{\bigl(w^{\mathcal{S}_k}_o \cdot \operatorname{ReLU}(W^{\mathcal{S}_k}_h\, x + b^{\mathcal{S}_k}_h) + b^{\mathcal{S}_k}_o\bigr)}_{= \mathcal{S}_k(x)}\Bigr].$$

In practice, since all groups share a single hidden layer, the per-group output weights (w^{C_k}_o scaled by alpha_k, etc.) and biases (b^{C_k}_o scaled by alpha_k, etc.) are concatenated into one output weight vector and one output bias scalar. The network is:

$$\mathcal{N}(x) = W_{\text{out}} \cdot \operatorname{ReLU}(W_{\text{hidden}}\, x + b_{\text{hidden}}) + b_{\text{out}}$$

where W_hidden is the vertical stack of all group weight matrices, b_hidden concatenates all group biases, W_out concatenates all coefficient-scaled output weights, and b_out = alpha_0 + sum of all coefficient-scaled group biases.

The resulting network has:
- **Width** W = sum of all hidden neurons across all basis functions
- **Depth** L = 1 (single hidden layer)
- **Parameters**: O(d . sum ||k||_1) weights, all bounded by 8 . max{1, |alpha_k|, |beta_k|}

The architecture is **independent of f** — only the 2 #I + 1 scalar parameters (alpha_0, alpha_k, beta_k) need to change when approximating different functions. This is the "transference principle" noted in the paper (Remark 5): the hidden-layer weights are fixed building blocks; only the output layer encodes f.

## Usage

```python
from basis.riesz import RieszBasisGenerator
from basis.riesz_network import coefficients_to_network
from algorithm.least_squares import LeastSquaresAlgorithm
from grid.generator.uniform_grid_generator import UniformGridGenerator
from solver.scipy_lstsq_solver import ScipyLstsqSolver

# 1. Define index set and build algorithm
index_set = [(k,) for k in range(1, 9)]    # K = 8
alg = LeastSquaresAlgorithm(
    basis_generator=RieszBasisGenerator(index_set),
    grid_generator=UniformGridGenerator(seed=42, multiplier_fun=lambda x: 200),
    solver=ScipyLstsqSolver(driver='gelsy'),
)

# 2. Fit
alg.fit(dim=1, scale=1, f=my_function)

# 3. Convert to ReLU network
gen = alg.basis_generator
alpha_0, coeff_dict = gen.coefficients_to_dict(alg.coeff)
net = coefficients_to_network(alpha_0, coeff_dict, dim=1)

# 4. Evaluate
import numpy as np
x = np.linspace(0, 1, 100).reshape(-1, 1)
y_net = net(x)              # numpy evaluation
torch_net = net.to_torch()  # PyTorch nn.Sequential
```

## What `network_to_coefficients` Implements (Backward Direction)

Given a depth-1, univariate ReLU network N on [0,1], `network_to_coefficients(net, max_freq)` recovers Riesz basis coefficients (α₀, α_k, β_k) such that

$$\mathcal{N}(x) = \alpha_0 + \sum_{k=1}^{K}\bigl[\alpha_k\,\mathcal{C}_k(x) + \beta_k\,\mathcal{S}_k(x)\bigr]$$

without solving a least-squares problem. The procedure has two steps, corresponding to **Equations (8)–(9)** in the paper.

### Step 1 — Network → Fourier coefficients (`network_to_fourier`)

A depth-1, d=1 ReLU network computes a continuous piecewise-linear (CPL) function on [0,1]. Its structure is fully determined by the hidden-layer weights and biases:

- Each neuron ReLU(w_j · x + b_j) has a kink (breakpoint) at x = −b_j / w_j.
- Collecting all kinks in [0,1] and sorting them gives the linear pieces.

The Fourier coefficients of a CPL function can be computed in **closed form**. On each piece [x_i, x_{i+1}] where f(x) = α_i x + β_i, the integrals

$$\int_{x_i}^{x_{i+1}} (\alpha_i x + \beta_i)\cos(2\pi n x)\,dx, \qquad \int_{x_i}^{x_{i+1}} (\alpha_i x + \beta_i)\sin(2\pi n x)\,dx$$

have elementary antiderivatives:

$$\int(\alpha x + \beta)\cos(\omega x)\,dx = \frac{\alpha\cos(\omega x)}{\omega^2} + \frac{(\alpha x + \beta)\sin(\omega x)}{\omega}$$

$$\int(\alpha x + \beta)\sin(\omega x)\,dx = \frac{\alpha\sin(\omega x)}{\omega^2} - \frac{(\alpha x + \beta)\cos(\omega x)}{\omega}$$

Summing these evaluations across all pieces gives exact Fourier coefficients a_0, a_1, …, a_K and b_1, …, b_K with no numerical quadrature error.

### Step 2 — Fourier → Riesz coefficients (`fourier_to_riesz`)

The Fourier series of the basis functions C and S are (Eq. 4):

$$\mathcal{C}(x) = \sum_{\substack{n=1 \\ n \text{ odd}}}^{\infty} \frac{8}{\pi^2 n^2}\cos(2\pi n x), \qquad \mathcal{S}(x) = \sum_{\substack{n=1 \\ n \text{ odd}}}^{\infty} \frac{8\,(-1)^{(n-1)/2}}{\pi^2 n^2}\sin(2\pi n x).$$

Since C_k(x) = C(kx), its Fourier contribution lands at frequencies n·k for odd n. The relationship between Fourier coefficients (a_m, b_m) and Riesz coefficients (α_k, β_k) is therefore a Dirichlet convolution over odd integers, invertible via the Möbius function μ:

$$\alpha_k = \frac{\pi^2}{8}\sum_{\substack{d \mid k \\ d \text{ odd}}} \frac{\mu(d)}{d^2}\,a_{k/d}$$

$$\beta_k = \frac{\pi^2}{8}\sum_{\substack{d \mid k \\ d \text{ odd}}} \frac{(-1)^{(d-1)/2}\,\mu(d)}{d^2}\,b_{k/d}$$

and α₀ = a₀. The Möbius function μ(d) is 0 if d has a squared prime factor, and (−1)^r if d is a product of r distinct primes. In practice only squarefree odd divisors contribute, so the sum has few terms.

### Roundtrip property

For networks produced by `coefficients_to_network`, the roundtrip

$$(\alpha_0, \alpha_k, \beta_k) \;\xrightarrow{\texttt{coefficients\_to\_network}}\; \mathcal{N} \;\xrightarrow{\texttt{network\_to\_coefficients}}\; (\hat\alpha_0, \hat\alpha_k, \hat\beta_k)$$

recovers the original coefficients to ~10⁻⁸ precision (limited only by floating-point arithmetic in the trigonometric evaluations and Möbius summation).

### Limitations

1. **Univariate only (d=1).** In d > 1, the linear regions of a ReLU network are polytopes in [0,1]^d. Computing multivariate Fourier integrals ∫_P f(x) cos(2πm·x) dx over each polytope analytically is a substantially harder problem (polyhedral integration). Additionally, the Möbius inversion generalises to a divisibility structure over lattice vectors in Z^d rather than simple integer division, requiring non-trivial bookkeeping.

2. **Depth 1 only.** Breakpoint extraction assumes each hidden neuron contributes exactly one kink. For deeper networks, composing piecewise-linear layers creates an exponentially complex structure — a depth-L, width-W network can have up to O(W^L) linear regions. Exact enumeration is expensive. *In practice this is not a limitation for the roundtrip use case*, since `coefficients_to_network` always produces depth-1 networks.

3. **Truncation at `max_freq`.** The Möbius inversion for α_k requires Fourier coefficients a_m for all m dividing k·n (odd n). If `max_freq` is too small, contributions from higher Fourier modes are missed. For exact roundtrip of a network built from frequencies up to K, using `max_freq ≥ K` suffices since all relevant Fourier energy is concentrated at frequencies ≤ K.

### Usage

```python
from basis.riesz_network import (
    coefficients_to_network,
    network_to_coefficients,
    network_to_fourier,
    fourier_to_riesz,
)

# Forward: coefficients → network
coeffs = {(1,): (0.5, -0.3), (2,): (0.7, 0.2), (3,): (-0.4, 0.6)}
net = coefficients_to_network(1.5, coeffs, dim=1)

# Backward: network → coefficients
alpha_0, recovered = network_to_coefficients(net, max_freq=50)
# alpha_0 ≈ 1.5, recovered[(1,)] ≈ (0.5, -0.3), etc.

# Or step by step:
a, b = network_to_fourier(net, max_freq=50)       # Step 1
alpha_0, recovered = fourier_to_riesz(a, b, max_k=3)  # Step 2
```

## Reference

C. Schneider, M. Ullrich, and J. Vybíral. *Nonlocal techniques for the analysis of deep ReLU neural network approximations.* arXiv:2504.04847, April 2025.

The construction is based on Lemma 5 (Section 3) and the Riesz basis properties established in [53]: C. Schneider and J. Vybíral. *A multivariate Riesz basis of ReLU neural networks.* Appl. Comput. Harmon. Anal. 68 (2024), 101605.
