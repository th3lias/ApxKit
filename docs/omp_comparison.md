# OMP vs LS / WLS Comparison

## Research question

Does OMP with a Tchebychev hyperbolic-cross basis outperform Least Squares (LS) and
Weighted Least Squares (WLS) on smooth Genz functions, given the same number of training
points?

---

## Algorithms

### Least Squares (LS)
- **Basis:** Clenshaw–Curtis Chebyshev polynomials on the Smolyak index set.
- **Grid:** N = 2 · M_LS random uniform points on [0, 1]^D (2× oversampled).
- **Fit:** Solves the overdetermined N × M_LS system A x = y.
- **Basis size:** M_LS = `calculate_num_points(D, scale)`.

### Weighted Least Squares (WLS)
- Same basis and same fit as LS.
- **Grid:** N = 2 · M_LS random Chebyshev-distributed points (arcsine measure).

### Orthogonal Matching Pursuit (OMP)
- **Basis:** Tchebychev polynomials T_k(x) = cos(k · arccos(x)) on the hyperbolic-cross
  index set HC(D, R) = { k ∈ ℕ₀^D : ∏_d max(1, k_d) ≤ R }.
- **Grid:** N = 2 · M_LS random Chebyshev-distributed points (same N as LS/WLS).
- **Fit:** Greedy sparse selection — picks at most s ≤ min(`num_iters`, M_OMP, N)
  basis functions via incremental Cholesky OMP.
- **Dictionary size:** M_OMP = |HC(D, R)|, auto-selected so that M_OMP ≥ M_LS.

---

## Parameters

| Symbol | Meaning |
|--------|---------|
| D | Dimension of the input space |
| scale | Depth parameter controlling both N and M_LS |
| N | Training points = 2 · calculate_num_points(D, scale) |
| M_LS | LS basis size = calculate_num_points(D, scale) = N / 2 |
| M_OMP | OMP dictionary size = \|HC(D, R)\| ≥ M_LS (auto-selected) |
| R | Hyperbolic-cross bandwidth (smallest R with \|HC(D, R)\| ≥ M_LS) |
| s | Effective OMP sparsity = number of selected basis functions ≤ min(num_iters, M_OMP, N) |
| num_iters | Maximum sparsity cap = 20 000 (matches SparseRecovery) |
| tol | Early-stopping threshold: stop when ‖residual‖₂ / √N < 1e-4 |

### Comparison design

Both LS and OMP see the same N training points (same scale, same 2× multiplier).
OMP's dictionary is at least as large as LS's basis (M_OMP ≥ M_LS), so OMP can
in principle recover any approximation that LS could produce. In practice OMP
selects far fewer than M_LS functions (sparsity), which is the hypothesis being tested.

The bandwidth R is determined automatically at fit time:
```
R = min { r ≥ 1 : |HC(D, r)| ≥ calculate_num_points(D, scale) }
```

---

## Feasibility: N, M and memory per (D, scale)

Memory = N · M_OMP · 8 bytes (float64 design matrix, materialised explicitly by ApxKit OMP).
Runs marked `!!!` exceed ~30 GB and are excluded from the small experiment.

| D | scale |        N |   M_LS | M_OMP |   R | mem (GB) |
|---|-------|----------|--------|-------|-----|----------|
| 2 |     1 |       10 |      5 |     8 |   2 |     0.00 |
| 2 |     2 |       26 |     13 |    17 |   4 |     0.00 |
| 2 |     3 |       58 |     29 |    31 |   7 |     0.00 |
| 2 |     4 |      130 |     65 |    70 |  14 |     0.00 |
| 2 |     5 |      290 |    145 |   150 |  27 |     0.00 |
| 2 |     6 |      642 |    321 |   322 |  52 |     0.00 |
| 2 |     7 |     1410 |    705 |   711 | 104 |     0.01 |
| 2 |     8 |     3074 |   1537 |  1537 | 205 |     0.04 |
| 3 |     1 |       14 |      7 |     8 |   1 |     0.00 |
| 3 |     2 |       50 |     25 |    32 |   3 |     0.00 |
| 3 |     3 |      138 |     69 |    86 |   6 |     0.00 |
| 3 |     4 |      354 |    177 |   177 |  11 |     0.00 |
| 3 |     5 |      882 |    441 |   459 |  22 |     0.00 |
| 3 |     6 |     2146 |   1073 |  1075 |  42 |     0.02 |
| 3 |     7 |     5122 |   2561 |  2591 |  84 |     0.11 |
| 3 |     8 |    12034 |   6017 |  6147 | 168 |     0.59 |
| 4 |     1 |       18 |      9 |    16 |   1 |     0.00 |
| 4 |     2 |       82 |     41 |    48 |   2 |     0.00 |
| 4 |     3 |      274 |    137 |   168 |   5 |     0.00 |
| 4 |     4 |      802 |    401 |   424 |   9 |     0.00 |
| 4 |     5 |     2210 |   1105 |  1193 |  18 |     0.02 |
| 4 |     6 |     5858 |   2929 |  3207 |  36 |     0.15 |
| 4 |     7 |    15074 |   7537 |  7737 |  70 |     0.93 |
| 4 |     8 |    37890 |  18945 | 19152 | 136 |     5.81 |
| 5 |     1 |       22 |     11 |    32 |   1 |     0.00 |
| 5 |     2 |      122 |     61 |   112 |   2 |     0.00 |
| 5 |     3 |      482 |    241 |   352 |   4 |     0.00 |
| 5 |     4 |     1602 |    801 |  1032 |   8 |     0.01 |
| 5 |     5 |     4866 |   2433 |  2592 |  15 |     0.10 |
| 5 |     6 |    13986 |   6993 |  7042 |  29 |     0.79 |
| 5 |     7 |    38626 |  19313 | 19628 |  56 |     6.07 |
| 5 |     8 |   103426 |  51713 | 51853 | 111 |    42.90 !!! |
| 6 |     1 |       26 |     13 |    64 |   1 |     0.00 |
| 6 |     2 |      170 |     85 |   256 |   2 |     0.00 |
| 6 |     3 |      778 |    389 |   448 |   3 |     0.00 |
| 6 |     4 |     2914 |   1457 |  1744 |   6 |     0.04 |
| 6 |     5 |     9730 |   4865 |  5696 |  12 |     0.44 |
| 6 |     6 |    30242 |  15121 | 17180 |  24 |     4.16 |
| 6 |     7 |    89378 |  44689 | 50220 |  48 |    35.91 !!! |

---

## Implementation notes

- `_find_hc_bandwidth(dim, target_m)` in `algorithm/omp.py` performs a binary search
  using the fast counting function `_hyp_cross_size` (no array allocation).
- The full N × M_OMP Tchebychev matrix is materialised as float64. For large (D, scale)
  this limits applicability; SparseRecovery avoids this via PyKeOps lazy evaluation.
- OMP uses incremental Cholesky updates (see https://ieeexplore.ieee.org/document/6333943).
  The algorithm is identical to SparseRecovery's `OMP.Tchebychev()` path.
