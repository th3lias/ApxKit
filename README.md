# A Toolbox for Polynomial Approximation

This repository contains a collection of polynomial approximation methodologies and [testing functions](https://www.sfu.ca/~ssurjano/integration.html). The toolbox is designed to be flexible and easy to use, allowing users to approximate functions using different polynomial bases and methods.

The toolbox already includes the following methods:
- Least Squares in various bases,
- [Smolyak's Algorithm](https://encyclopediaofmath.org/wiki/Smolyak_algorithm) with the [Tasmanian](https://github.com/ORNL/TASMANIAN) library.

We are actively working on adding more methods and functionality such as testing possibilities for least squares on multi-node clusters and Wavelet bases.

The library is entirely written in [Python](https://www.python.org) and is designed to be modular, allowing users to easily extend its functionality. We tried to keep the code well-documented, and encourage contributions. Generally, we try to follow the [PEP8](https://peps.python.org/pep-0008/) style guide for Python code and an object-oriented approach. We welcome any feedback or suggestions for improvements.

## Methodologies 🧮

As mentioned above, the toolbox currently includes some standard methods for polynomial approximation. Here, we briefly want to give an overview on our implementation. In the following, we consider the problem of approximating a function $f$ on the unit cube $[0,1]^d$ or $[-1,1]^d$. The approximation is given by a polynomial of degree $n$, i.e. $p_\star \in P^n$, which solves the problem

$$\min_{p \in P^n} \left\lVert f - p \right\rVert$$

for some norm $\left\lVert \cdot \right\rVert$.

### Least Squares 🟧🟩

The least squares method is a standard approach for polynomial approximation. In general, least-squares tries to minimise the seminorm $\left\lVert \cdot \right\rVert_X$, defined by

$$\left\lVert f - p \right\rVert_X^2 = \frac{1}{\lvert X \rvert} \sum_{x \in X} \left\lvert f(x) - p(x)\right\rvert^2.$$

In this context, $X$ is a (possibly random) set of points in the domain. To actually find the minimising polynomial, we consider the overdetermined linear system

$$A \cdot z = b$$

for a matrix $A \in \mathbb{R}^{m \times n}$, a vector $z \in \mathbb{R}^n$ of coefficients and a vector $b \in \mathbb{R}^{m}$ of function values. The matrix $A$ is constructed from the basis functions evaluated at the points in $X$, i.e. $A_{ij} = \phi_j(x_i)$, where $\phi_j$ are the basis functions and $x_i$ are the points in $X$. In our case, various basis functions are implemented. The vector $b$ is given by $b_i = f(x_i)$ and the vector $z$ contains the coefficients of the polynomial in the chosen basis. The system can then be written as

$$p_{\star} = \sum_{j=0}^{n} z_{\star_j} \phi_j$$

where $z_{\star_j}$ is the least-squares solution to above's system.

To actually compute the solution to the above system (in a numerically stable way) there exist various methods such as QR decomposition or Singular Value Decomposition (SVD). In our implementation, we use [Scipy](https://scipy.org)'s [lstsq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html) method which uses [Lapack](https://www.netlib.org/lapack/)'s [gelsy](https://www.netlib.org/lapack/explore-html/dc/d8b/group__gelsy.html) driver in the backend.

Even though, least squares is an extremely simple algorithm conceptually, it is of quite large importance in practice and theory, see

- [On the worst-case error of least squares algorithms for L2-approximation with high probability](https://arxiv.org/abs/2003.11947),
- [Sampling projections in the uniform norm](https://arxiv.org/abs/2401.02220)
- [Least Squares Optimization: from Theory to Practice](https://arxiv.org/abs/2002.11051).

### Smolyak's Algorithm 🤯🚀

Smolyak's algorithm is a method for constructing high-dimensional polynomial approximations using a sparse grid. The idea is to combine low-dimensional polynomial approximations in a way that reduces the number of points needed in total. This is particularly useful for high-dimensional problems, where the curse of dimensionality makes traditional methods infeasible. For this, consider the standard one-dimensional polynomial **interpolation**

$$p = \sum_{i=0}^{n} f(x_i) \ell_i$$

where $\ell_i$ are the Lagrange basis functions, i.e. $\ell_i(x_j) = \delta_{ij}$. In higher dimensions, we may formulate the interpolating basis $\ell_i$ as

$$\ell_i(\xi) = \bigotimes_{j=1}^{d} \ell_{i_j}(\xi_j) = \ell_{i_1}(\xi_1) \cdot \ell_{i_2}(\xi_2) \cdots \ell_{i_d}(\xi_d) = \prod_{j=1}^{d} \prod_{\substack{k=1 \\ k \neq i_j}}^{n_j} \frac{\xi_j - x_j^k}{x_j^{i_j} - x_j^k}$$

where $\xi = (\xi_1, \ldots, \xi_d)$ and
$\{x_j^k \mid 1 \leq k \leq n_j,\ 1 \leq j \leq d\}$
are the points used for interpolation and $i = (i_1, \dots, i_d) \in \mathbb{N}^d$ is a multiindex. There are many different ways to choose the points $x_j^k$. For example, we mostly use Chebyshev points, which are given by

$$x_j^k = - \cos \left( \frac{\pi (k-1)}{n_j - 1} \right) \quad \text{for } k = 1, \ldots, n_j.$$

The multiindex $(i_1, \ldots, i_d)$ is used to restrict the polynomial degree in each dimension. In fact, Smolyak's construction upper-bounds the total degree of the polynomial by $n = \sum_{j=1}^{d} i_j \leq q$ for some $q > d$. Hence, $q$ is a resolution parameter (or scale). To make this algorithm efficient, one seeks to use overlapping point sets. This may be achieved by combining a doubling strategy for the number $n_j$, i.e. $n_j = 2^{j-1}+1$ for $j = 2, \dots, d$ and $n_1=1$ and using the Chebyshev points as above. Then, Smolyak's algorithm can be written in a recursive style as

$$A(q, d) = A(q-1, d) + \sum_{\lVert i \rVert_1 = q} \bigotimes_{j=1}^d \Delta_{i_j}$$

where $\Delta_{i_j}$ is the difference operator
$I_{i_{j}} - I_{i_{j}-1}$ and $I_{i_{j}}$
is the interpolation operator
$I_{i_{j}}(f) = \sum_{k=1}^{n_{i_j}} f\left( x^k \right) \ell_{i_j}$
with $I_0 = 0$.

There is a large body of literature on Smolyak's algorithm. We refer to

- the landmark paper [High dimensional polynomial interpolation on sparse grids](https://link.springer.com/article/10.1023/A:1018977404843),
- [Smolyak's algorithm: A powerful black box for the acceleration of scientific computations](https://arxiv.org/abs/1703.08872) and
- [User Manual: TASMANIAN Sparse Grids](https://mkstoyanov.github.io/tasmanian_aux_files/docs/TasmanianMathManual.pdf).

for further information.

## How to cite 📝

This toolbox was developed as part of a research project at the [Johannes Kepler University Linz](https://www.jku.at), Austria, and is intended for educational and research purposes. If you intend to use this toolbox for your own research, please cite the following paper:

```bibtex
@article{EgglMindlbergerUllrich2025,
  title={Sparse grids vs. random points for high-dimensional polynomial approximation},
  author={Eggl, Jakob and Mindlberger, Elias and Ullrich, Mario},
  journal={//},
  year={2025}
}
```

If you have questions, feedback or want to reach out for any other reason, please contact us at

    jakob (dot) eggl (at) jku (dot) at,
    elias (dot) mindlberger (at) jku (dot) at,
    mario (dot) ullrich (at) jku (dot) at.