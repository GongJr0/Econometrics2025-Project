import numpy as np
import numpy.typing as npt
import sympy as sp  # type: ignore
import pandas as pd
from typing import Union, Literal

from .error_functions import r2, r2_adj, rmse, mape
from .fit_data import ErrorMetrics, FitResults
from .stat_tests import ADF, BP, SW, BG, F_TEST, HC

class OLS:
    """Ordinary Least Squares (OLS) Regression Model"""

    def __init__(
        self, X: Union[pd.DataFrame, np.float64], y: Union[pd.Series, np.ndarray]
    ):
        assert isinstance(
            X, (pd.DataFrame, np.ndarray)
        ), "X must be a DataFrame or ndarray"
        assert isinstance(y, (pd.Series, np.ndarray)), "y must be a Series or ndarray"

        assert (
            X.shape[0] == y.shape[0]
        ), f"Row count mismatch between fetare and target set. {X.shape[0]=}, {y.shape[0]=}"

        assert X.dtype == float or X.dtype == int, "All X columns must be numeric."
        assert y.dtype == float or y.dtype == int, "y values must be numeric."

        self.X = self.get_named_X(X)
        self.y = y

        self._n_cols: int | None = None
        self._n_obs: int | None = None

    def fit(self, diagnosis_alpha: float = 0.05, diagnosis_trend: Literal["c", "ct", "ctt", "n"] = "c", **kwargs) -> FitResults:
        X = self.X.values
        X_raw = X.copy()
        X = np.array([[1, *row] for row in X], dtype=np.float64)
        if isinstance(self.y, pd.Series):
            y = self.y.values
        else:
            y = self.y

        XT = X.T
        XT_X = XT @ X

        betas = np.linalg.inv(XT_X) @ XT @ y
        self.betas = betas
        

        y_hat = X @ betas
        resid = y - y_hat
        
        XT_e = np.sum(XT @ resid)

        SSR = np.sum((y_hat - np.mean(y))**2)
        SSE = np.sum((resid)**2)
        
        dfn = X.shape[1]
        dfd = X.shape[0]-X.shape[1]-1
        F = ((SSR/dfn) / (SSE/dfd))
        
         

        err = ErrorMetrics(
            r2=round(r2(y, y_hat), 4),
            r2_adj=round(r2_adj(y, y_hat, X.shape[1]), 4),
            rmse=round(rmse(y, y_hat), 4),
            mape=round(mape(y, y_hat), 4),
        )

        BGN = lambda x: BG(resid, x, diagnosis_alpha)

        
        
        f_test = F_TEST(F, dfn, dfd, diagnosis_alpha)
        hc = HC(y, X, diagnosis_alpha)
        heteroske = BP(X_raw, y, diagnosis_alpha)
        stationarity = ADF(resid, diagnosis_trend, diagnosis_alpha)
        autocorr = [BGN(i) for i in range(1, 9)]  # BG tests for lags 1 to 8 (2 years quarterly)
        normality = SW(resid, diagnosis_alpha)

        return FitResults(
            fitted_values=y_hat,
            resid=resid,
            XT_e=XT_e,
            F_test=f_test,
            HC_test=hc,
            beta=betas,
            coefs=betas[1:],
            intercept=betas[0],
            error=err,
            resid_heteroske=heteroske,
            resid_autocorr=autocorr,
            resid_stationarity=stationarity,
            resid_normality=normality
        )
    
    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> npt.NDArray[np.float64]:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        X = np.column_stack([np.ones(X.shape[0]), X])

        return X@self.betas

    # ================= Symbolic Representation =================
    @staticmethod
    def get_named_X(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """If X is a DataFrame, return it as is; otherwise, convert ndarray to DataFrame with named columns X_{1:n}."""
        if isinstance(X, pd.DataFrame):
            return X

        n_cols = X.shape[1]
        cols = [f"X_{i}" for i in range(1, n_cols + 1)]

        return pd.DataFrame(data=X, columns=cols)

    def get_symbols(self) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
        """Returns X_i and beta_i symbols. Exlcudes beta_0 (intercept)"""
        vars = self.X.columns
        n_cols = self.n_cols
        cols = self.X.columns

        var_sym = []
        beta_sym = []

        y = sp.Symbol("y")
        beta_0 = sp.Symbol("beta_0")
        for i in range(1, n_cols + 1):
            var_sym.append(sp.Symbol(cols[i - 1]))
            beta_sym.append(sp.Symbol(f"beta_{i}"))

        return var_sym, beta_sym

    def get_equation(self) -> sp.Eq:
        """Returns the OLS regressor in equation form. (Only for visualization purposes.
        This property is not used in solution or prediction methods)"""

        var_sym, beta_sym = self.get_symbols()
        beta_0 = sp.Symbol("beta_0")
        y = sp.Symbol("y")

        expr = beta_0 + sum([x * y for x, y in zip(beta_sym, var_sym)])

        return sp.Eq(y, expr, evaluate=False)

    def get_matrix(self, n_rows=5):
        X = self.X.head(n_rows).copy().values
        y = sp.Matrix(self.y[:n_rows].values)

        _, beta_sym = self.get_symbols()
        beta_0 = sp.Symbol("beta_0")

        # Add 1s to X rows for intercept
        X = np.array([[1, *row] for row in X])

        X_mat = sp.Matrix(X)
        beta_vec = sp.Matrix([beta_0, *beta_sym])

        expr = X_mat * beta_vec
        return sp.Eq(y, expr, evaluate=False)

    # ================= Properties =================
    @property
    def n_cols(self) -> int:
        if not self._n_cols:
            self._n_cols = self.X.shape[1]

        return self._n_cols

    @property
    def n_obs(self) -> int:
        if not self._n_obs:
            self._n_obs = self.X.shape[0]

        return self._n_obs
