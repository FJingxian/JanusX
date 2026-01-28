from typing import Union
import numpy as np
from scipy.optimize import minimize

def FaSTREML(
    loglbd: float,
    U1tsnp: Union[np.ndarray,None],
    U2tsnp:Union[np.ndarray,None],
    U1ty: np.ndarray,
    U2ty: np.ndarray,
    U1tx: np.ndarray,
    U2tx: np.ndarray,
    S: np.ndarray
) -> float:
    """
    Restricted Maximum Likelihood (REML) for variance components.

    Parameters
    ----------
    theta : Sequence[float]
        Variance components for each covariance term in SIGMAlist.
    y : np.ndarray
        Response vector with shape (n, 1).
    X : np.ndarray
        Fixed-effect design matrix with shape (n, p).
    SIGMAlist : Sequence[np.ndarray]
        Covariance matrices for random effects plus residual.

    Returns
    -------
    float
        REML objective value (lower is better for minimize).
    """
    lbd = 10**loglbd
    v1 = S+lbd
    v1_inv = 1 / v1
    v2 = lbd
    v2_inv = 1 / lbd
    if (U1tsnp is None) ^ (U2tsnp is None):
        raise ValueError("U1tsnp and U2tsnp must be both provided or both None.")
    if U1tsnp is not None and U2tsnp is not None:
        U1tx = np.column_stack([U1tx, U1tsnp])
        U2tx = np.column_stack([U2tx, U2tsnp])

    n = U1ty.shape[0] + U2ty.shape[0]
    p = U1tx.shape[1]
    k = v1.size

    XTV_invX = (U1tx.T * v1_inv) @ U1tx + v2_inv * (U2tx.T @ U2tx)
    XTV_invy = (U1tx.T * v1_inv) @ U1ty + v2_inv * (U2tx.T @ U2ty)

    beta = np.linalg.solve(XTV_invX, XTV_invy)
    r1: np.ndarray = U1ty - U1tx @ beta
    r2: np.ndarray = U2ty - U2tx @ beta

    rTV_invr = np.sum(v1_inv * np.ravel(r1) ** 2) + v2_inv * np.sum(np.ravel(r2) ** 2)
    
    log_detV = np.sum(np.log(v1)) + (n - k) * np.log(v2)
    sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
    total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX

    # Constant term (matches your original expression)
    c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
    return total_log / 2.0 - c
