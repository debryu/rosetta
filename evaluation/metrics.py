"""
Metrics for evaluating source recovery in nonlinear ICA.

Mean Correlation Coefficient (MCC):
  Standard metric in ICA literature. Computes the absolute Pearson correlation
  matrix between estimated latents z and ground-truth sources s, then finds
  the optimal permutation (via Hungarian algorithm) that maximises the sum of
  diagonal correlations. MCC = mean of those optimal correlations.
  MCC = 1.0 → perfect recovery; MCC ≈ 0.0 → chance.

Linear probe R²:
  Fit an ordinary linear regression from z to each source factor independently.
  R² measures how much of each factor's variance is linearly captured by z.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def compute_mcc(z_hat: np.ndarray, s_true: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Mean Correlation Coefficient between estimated and true sources.

    Args:
        z_hat:  (N, latent_dim)  estimated latent codes
        s_true: (N, n_factors)   ground-truth source values

    Returns:
        mcc:        scalar mean correlation over optimal assignment
        assignment: (n_factors,) array of z indices assigned to each factor
    """
    n_factors = s_true.shape[1]
    latent_dim = z_hat.shape[1]

    # Pearson absolute correlation matrix: C[i, j] = |corr(z_i, s_j)|
    C = np.zeros((latent_dim, n_factors))
    for i in range(latent_dim):
        for j in range(n_factors):
            c = np.corrcoef(z_hat[:, i], s_true[:, j])[0, 1]
            C[i, j] = abs(c) if not np.isnan(c) else 0.0

    # Hungarian algorithm: maximise sum of selected entries
    row_ind, col_ind = linear_sum_assignment(-C)   # minimise negative = maximise positive

    # Extract the n_factors assignments (one z dimension per source factor)
    assignment = np.zeros(n_factors, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if c < n_factors:
            assignment[c] = r

    mcc = float(C[row_ind, col_ind].mean())
    return mcc, assignment


def linear_probe_r2(z_hat: np.ndarray, s_true: np.ndarray) -> np.ndarray:
    """Fit a linear regressor from z to each source factor independently.

    Args:
        z_hat:  (N, latent_dim)
        s_true: (N, n_factors)

    Returns:
        r2: (n_factors,) array of R² scores, one per factor
    """
    scaler_z = StandardScaler().fit(z_hat)
    z_scaled = scaler_z.transform(z_hat)

    n_factors = s_true.shape[1]
    r2 = np.zeros(n_factors)
    for j in range(n_factors):
        reg = LinearRegression().fit(z_scaled, s_true[:, j])
        r2[j] = reg.score(z_scaled, s_true[:, j])
    return r2
