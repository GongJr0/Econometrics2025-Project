import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.stats import t
from typing import Literal
from .fit_data import StatsTest
from .tau_coefs import mackinnon_p
from .sw_coefs import swilk
from .error_functions import r2
from math import gamma
from mpmath import gammainc
import warnings


class AR1:
    def __new__(cls, X, trend: Literal["c", "ct", "ctt", "n"] = "c") -> float64:
        return cls._dispatch(X, trend)

    @classmethod
    def _dispatch(cls, X, trend: Literal["c", "ct", "ctt", "n"]) -> NDArray[float64]:
        """Calculate the AR(1) coefficient for the given data using an iterative solver.

        Args:
            X (np.ndarray): 1D array of data points.
            trend (Literal["c", "ct", "n"]): Type of trend to include.
                "c" for constant, "ct" for constant and trend, "n" for no trend.
        Returns:
            float: AR(1) coefficient.
        """
        assert len(X)>=2, "Input array must have at least two elements."
        assert trend in ("c", "ct", "ctt", "n"), "Invalid trend type."

        match trend:
            case "c":
                return cls.c(X)
            case "ct":
                return cls.ct(X)
            case "ctt":
                return cls.ctt(X)
            case "n":
                return cls.n(X)
            case _:
                raise ValueError("Invalid trend type.")

    @staticmethod
    def n(X) -> NDArray[float64]:
        y = X[1:]
        x = X[:-1].reshape(-1, 1)

        xt = x.T
        beta: NDArray[float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta

    @staticmethod
    def c(X) -> NDArray[float64]:
        y = X[1:]
        x = np.column_stack([np.ones(len(y)), X[:-1]])
        xt = x.T
        beta: NDArray[float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta

    @staticmethod
    def ct(X) -> NDArray[float64]:
        y = X[1:]
        t = np.arange(1, len(X))
        x = np.column_stack([np.ones(len(y)), t, X[:-1]])
        xt = x.T
        beta: NDArray[float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta
    
    @staticmethod
    def ctt(X) -> NDArray[float64]:
        y = X[1:]
        t = np.arange(1, len(X))
        t2 = t**2
        x = np.column_stack([np.ones(len(y)), t, t2, X[:-1]])
        xt = x.T
        beta: NDArray[float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta


def ADF(
    X: NDArray[float64],
    trend: Literal["c","ct","ctt","n"]="c",
    alpha: float=0.05,
    lags: int=0,
) -> StatsTest:

    X = np.asarray(X, dtype=float64).squeeze()
    assert len(X) >= 3 + lags
    assert trend in ("c","ct","ctt","n")
    assert lags >= 0

    

    n = len(X)
    n_trend = {
        "n": 0,
        "c": 1,
        "ct": 2,
        "ctt": 3
    }[trend]

    if lags == 0:
        l = round(12*(n/100)**(1/4))  # heuristic from statsmodels' ADF implementation
        lags = min(l, n//2-n_trend-1)
        lags = max(lags, 0)

    dX = np.diff(X)
    start = lags

    y = dX[start:]
    x_level = X[start:-1]

    cols = []
    if trend in ("c","ct","ctt"):
        cols.append(np.ones_like(y))
    if trend in ("ct","ctt"):
        t = np.arange(start+1, n, dtype=float64)
        cols.append(t)
    if trend == "ctt":
        t = np.arange(start+1, n, dtype=float64)
        cols.append(t**2)

    cols.append(x_level)

    for i in range(1, lags+1):
        cols.append(dX[start-i:-(i)])

    Xreg = np.column_stack(cols)

    beta_hat, *_ = np.linalg.lstsq(Xreg, y, rcond=None)
    resid = y - Xreg @ beta_hat

    k = Xreg.shape[1]
    dof = len(y) - k
    s2 = (resid @ resid) / dof

    XtX_inv = np.linalg.inv(Xreg.T @ Xreg)
    se = np.sqrt(np.diag(s2 * XtX_inv))

    gamma_idx = len(cols) - (lags + 1)
    adf_stat = beta_hat[gamma_idx] / se[gamma_idx]

    p_value = mackinnon_p(float(adf_stat), trend)

    return StatsTest(
        reject=p_value<alpha,
        pval=float(p_value),
        test_stat=adf_stat,
        stat_name="ADF Test (T-Statistic)"
    )


def BP(X: NDArray[float64], 
       y: NDArray[float64],
       alpha:float64 = 0.05) -> StatsTest:

    ddof = X.shape[1]

    X = np.column_stack([np.ones(X.shape[0]), X])

    XT = X.T
    XT_X = XT@X

    beta = np.linalg.inv(XT_X) @ XT @ y

    fitted = X@beta
    e2 = (y - fitted)**2

    sigma_hat2 = sum(e2)/len(e2)
    g = e2/sigma_hat2

    aux_beta = np.linalg.inv(XT_X) @ XT @ g
    aux_fit = X@aux_beta
    aux_r2 = r2(g, aux_fit)

    chi_stat = len(g)*aux_r2
    pval = gammainc(ddof/2, 0, chi_stat/2)/gamma(ddof/2)  # Chi^2(X, ddof) CDF

    return StatsTest(
        reject=pval<alpha,
        pval=float(pval),
        test_stat=float(chi_stat),
        stat_name="Breusch-Pagan Test (Chi^2 Statistic)"
    )
    

def SW(X, alpha = 0.05) -> StatsTest:
    X = np.ravel(X).astype(float64)
    n = len(X)

    if n< 3:
        raise ValueError("Shapiro-Wilk test requires N>3")
    
    a = np.zeros(n//2, dtype=float64)

    y = np.sort(X)
    y -= X[n//2]  # approx median

    w, pw, ifault = swilk(y, a, 0)
    if ifault not in [0,2]:
        warnings.warn("Input data has range zero. "
                        "Results may be inaccurate")

    if n > 5000:
        warnings.warn("For N > 5000, computed p-value "
                      f"may not be accurate. Current N is {n}.")
    
    return StatsTest(
        reject=pw<alpha,
        pval=pw,
        test_stat=w,
        stat_name="Shapiro-Wilk Test (Approximated Z-Statistic)"
    )