import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union


class StandardScaler:
    """Stadnard (Z) Scaler implementation"""

    def __init__(self) -> None:

        # Variables populated at .fit()
        self.is_fitted: bool = False
        self.n_col: int = 0

        self.sigma: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.X_bar: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

    def fit(self, X: Union[pd.DataFrame, npt.NDArray[np.float64]]) -> None:
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_col = X.shape[1]

        self.sigma = np.zeros((n_col,), dtype=np.float64)
        self.X_bar = np.zeros((n_col,), dtype=np.float64)
        self.n_col = n_col

        for i in range(n_col):
            self.sigma[i] = np.sqrt(np.var(X[:, i]))
            self.X_bar[i] = np.mean(X[:, i])

        self.sigma = np.where(self.sigma == 0, 1e-12, self.sigma)  # Avoid DIV/0

        self.is_fitted = True

    def transform(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        assert (
            self.is_fitted
        ), "Scaler is not fitted. Please run .fit before calling .transform or use .fit_transform directly."
        assert (
            X.shape[1] == self.n_col
        ), f"Input does not have the same column count as fitted array. Cols at fit: {self.n_col}. {X.shape[1]=}"

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_bar = self.X_bar
        sigma = self.sigma

        out = np.zeros_like(X, dtype=np.float64)

        def scale(
            X: npt.NDArray[np.float64],
            X_bar: np.float64,
            sigma: np.float64,
        ) -> npt.NDArray[np.float64]:
            return (X - X_bar) / sigma

        for i in range(self.n_col):
            z = scale(X[:, i], X_bar[i], sigma[i])
            out[:, i] = z

        return out

    def fit_transform(
        self,
        X: npt.NDArray[np.float64] | pd.DataFrame,
    ) -> npt.NDArray[np.float64]:
        
        self.fit(X)
        return self.transform(X)

    def inverse_scale(
        self, Z: Union[pd.DataFrame, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        assert (
            self.is_fitted
        ), "Scaler is not fitted. Please run .fit before calling .transform or use .fit_transform directly."
        assert (
            Z.shape[1] == self.n_col
        ), f"Input does not have the same column count as fitted array. Cols at fit: {self.n_col}. {Z.shape[1]=}"

        if isinstance(Z, pd.DataFrame):
            Z = Z.values

        X_bar = self.X_bar
        sigma = self.sigma

        out = np.zeros_like(Z, dtype=np.float64)

        def inv_scale(
            Z: npt.NDArray[np.float64],
            X_bar: np.float64,
            sigma: np.float64,
        ) -> Union[pd.Series[np.float64], npt.NDArray[np.float64]]:
            return Z * sigma + X_bar

        for i in range(self.n_col):
            X = inv_scale(Z[:, i], X_bar[i], sigma[i])
            out[:, i] = X

        return out
