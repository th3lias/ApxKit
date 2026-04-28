import hashlib

import numpy as np


class UniformNumberGenerator:
    """Deterministically seeded generator for uniform random numbers with caching."""

    def __init__(self, seed=42):
        self._reshuffle_count = 0
        self.seed = seed
        self._cache = dict()

    def _generate_seed(self, n_points: int, dim: int, lower_bound: float = 0., upper_bound: float = 1.) -> int:
        key = f"{self.seed}-{self._reshuffle_count}-{n_points}-{dim}-{lower_bound}-{upper_bound}"
        hash_digest = hashlib.sha256(key.encode()).hexdigest()
        return int(hash_digest[:16], 16) % (2 ** 32)

    def get_random_numbers(self, n_points: int, dim: int, lower_bound: float = 0.,
                           upper_bound: float = 1.) -> np.ndarray:
        key = (n_points, dim, lower_bound, upper_bound)
        if key not in self._cache:
            np.random.seed(self._generate_seed(*key))
            self._cache[key] = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_points, dim))
        return self._cache[key].squeeze()

    def reshuffle(self):
        """Invalidate cache and increment reshuffle counter for new draws."""
        self._reshuffle_count += 1
        self._cache.clear()
