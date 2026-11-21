"""Two empirical random-variable generators with explicit mathematical contracts.

X = danceability, Y = valence. Both are scaled to [0, 1].
No prediction of a specific song is attempted; the result simulates new observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ("danceability", "valence")


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(COLS).issubset(frame.columns):
        raise ValueError("Expected columns danceability and valence")
    result = frame.loc[:, COLS].copy()
    if len(result) < 20 or result.isna().any().any():
        raise ValueError("At least 20 complete observations are required")
    a = result.to_numpy(dtype=float)
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("Values must be finite and belong to [0, 1]")
    return result


def empirical_quantile(values: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Inverse empirical CDF: Q_n(u) = X_(floor(n*u)+1), 0 <= u < 1."""
    v = np.sort(np.asarray(values, dtype=float))
    u = np.asarray(uniforms, dtype=float)
    if not len(v) or not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
        raise ValueError("Nonempty values and uniforms in [0,1) are required")
    return v[np.floor(u * len(v)).astype(int)]


def conditional_generator(frame: pd.DataFrame, n: int, seed: int = 42,
                          bins: int = 5) -> pd.DataFrame:
    """Draw X empirically, then draw Y from its empirical conditional CDF.

    The condition is X's observed quantile bin. Independent U(0,1)
    variates are used in both steps. This preserves binned dependence,
    not every detail of the original continuous copula.
    """
    base = _validate(frame)
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 1_000_000:
        raise ValueError("n must be an integer in [1, 1_000_000]")
    if not isinstance(bins, (int, np.integer)) or bins < 2 or bins > 10:
        raise ValueError("bins must be an integer between 2 and 10")
    x = base["danceability"].to_numpy()
    y = base["valence"].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Not enough distinct danceability values for bins")
    classes = np.searchsorted(edges[1:-1], x, side="right")
    rng = np.random.default_rng(seed)
    x_gen = empirical_quantile(x, rng.random(n))
    c_gen = np.searchsorted(edges[1:-1], x_gen, side="right")
    y_gen = np.empty(n, dtype=float)
    for j in np.unique(c_gen):
        use = c_gen == j
        pool = y[classes == j]
        if not len(pool):
            raise ValueError("Empty conditional bin; try fewer bins")
        y_gen[use] = empirical_quantile(pool, rng.random(int(use.sum())))
    return pd.DataFrame({"danceability": x_gen, "valence": y_gen})


def independent_generator(frame: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Control: matching marginals but deliberately removing X-Y dependence."""
    base = _validate(frame)
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 1_000_000:
        raise ValueError("n must be an integer in [1, 1_000_000]")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: empirical_quantile(base[c], rng.random(n)) for c in COLS})
