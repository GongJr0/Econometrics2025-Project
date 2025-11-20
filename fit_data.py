from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt

# ================= Data Classes =================
@dataclass
class ErrorMetrics:
    r2: float
    r2_adj: float
    rmse: float
    mape: float

    def __repr__(self) -> str:
        return "\n".join([f"{k.upper()}: {v}" for k, v in asdict(self).items()])


@dataclass
class StatsTest:
    reject: bool
    pval: float
    test_stat: float
    stat_name: str

    def __repr__(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in asdict(self).items()])


@dataclass
class FitResults:
    fitted_values: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]

    error: ErrorMetrics

    resid_heteroska: StatsTest
    resid_stationarity: StatsTest

    def __repr__(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in asdict(self).items()])