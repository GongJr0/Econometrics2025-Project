import numpy as np
import numpy.typing as npt
from scipy.stats import t
from typing import Literal
from fit_data import StatsTest

class AR1:
    def __call__(self, X, trend: Literal["c", "ct", "n"] = "c") -> np.float64:
        """Calculate the AR(1) coefficient for the given data using an iterative solver.

        Args:
            X (np.ndarray): 1D array of data points.
            trend (Literal["c", "ct", "n"]): Type of trend to include.
                "c" for constant, "ct" for constant and trend, "n" for no trend.
        Returns:
            float: AR(1) coefficient.
        """
        assert len(X)>=2, "Input array must have at least two elements."
        assert trend in ("c", "ct", "n"), "Invalid trend type."

        match trend:
            case "c":
                return self.c(X)
            case "ct":
                return self.ct(X)
            case "n":
                return self.n(X)
            case _:
                raise ValueError("Invalid trend type.")

    @staticmethod
    def n(X) -> np.float64:
        y = X[1:]
        x = X[:-1].reshape(-1, 1)

        xt = x.T
        beta: npt.NDArray[np.float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta.tolist()[0]

    @staticmethod
    def c(X) -> np.float64:
        y = X[1:]
        x = np.column_stack([np.ones(len(y)), X[:-1]])
        xt = x.T
        beta: npt.NDArray[np.float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta.tolist()[1]

    @staticmethod
    def ct(X) -> np.float64:
        y = X[1:]
        t = np.arange(1, len(X))
        x = np.column_stack([np.ones(len(y)), t, X[:-1]])
        xt = x.T
        beta: npt.NDArray[np.float64] = np.linalg.inv(xt @ x) @ xt @ y
        return beta.tolist()[2]


def ADF(X, trend: Literal["c", "ct", "n"] = "c", alpha: float = 0.05) -> StatsTest:
    """Perform Augmented Dickey-Fuller test to check for stationarity.

    Args:
        X (np.ndarray): 1D array of residuals.
        trend (Literal["c", "ct", "n"]): Type of trend to include.
            "c" for constant, "ct" for constant and trend, "n" for no trend.
        alpha (PVal): Significance level for the test.

    Returns:
        float: p-value of the test.
    """
    assert len(X) >= 3, "Input array must have at least three elements."
    assert trend in ("c", "ct", "n"), "Invalid trend type."

    sigma_x = np.std(X, ddof=1)
    n = len(X)

    se_X = sigma_x * np.sqrt(n)
    beta = AR1()(X, trend=trend)
    adf_stat = beta/se_X
    # Approximate p-value using t-distribution
    p_value = t.cdf(adf_stat, df=n-1)

    return StatsTest(
        reject=p_value.reject,
        pval=float(p_value),
        test_stat=adf_stat,
        stat_name="ADF Test (T-Statistic)"
    )

