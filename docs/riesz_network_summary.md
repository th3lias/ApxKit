# From Riesz Basis Coefficients to ReLU Networks

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

## Reference

C. Schneider, M. Ullrich, and J. Vybíral. *Nonlocal techniques for the analysis of deep ReLU neural network approximations.* arXiv:2504.04847, April 2025.

The construction is based on Lemma 5 (Section 3) and the Riesz basis properties established in [53]: C. Schneider and J. Vybíral. *A multivariate Riesz basis of ReLU neural networks.* Appl. Comput. Harmon. Anal. 68 (2024), 101605.
