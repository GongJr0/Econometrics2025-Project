from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from mpmath import erfinv
from scipy.optimize import minimize
import numpy.typing as npt
from typing import Literal

import matplotlib.pyplot as plt


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
    
    # Complete beta vector
    beta: npt.NDArray[np.float64]

    # Coefficients of the variables used in fitting
    coefs: npt.NDArray[np.float64]

    intercept: np.float64

    XT_e: np.float64
    error: ErrorMetrics

    resid_heteroske: StatsTest
    resid_stationarity: StatsTest
    resid_autocorr: list[StatsTest]
    resid_normality: StatsTest

    def qq(self,
           *,
           line: Literal["r", "q", "45"] | None = None,
           nsim: int = 1000,
           band: float = 0.95,
           seed: int | None = None,
        ):

        resid = np.asarray(self.resid, dtype=np.float64)
        n = len(resid)

        x, y = _qq_points(resid)

        sup = lab = None
        has_line = line is not None
        if line == "q":
            y1, y3 = np.percentile(y, [25, 75])
            x1, x3 = _std_norm_inv_cdf_vec(np.array([0.25, 0.75]))
            m = (y3 - y1) / (x3 - x1)
            b = y1 - m*x1
            sup = (x, m*x + b)
            lab = "Quartile Fit"

        elif line == "r":
            m, b = np.polyfit(x, y, 1)
            sup = (x, m*x + b)
            lab = "Regression Line"

        elif line == "45":
            mu = y.mean()  # approx 0
            sig = y.std(ddof=1)  # unb. stdev
            sup = (x, mu + sig*x)
            lab = r"$\bar{y} + S_y x_i$"

        else:
            has_line = False

        # ---- Simulation envelope ----
        rng = np.random.default_rng(seed)

        mu_hat = resid.mean()
        sig_hat = resid.std(ddof=1)

        sims = rng.standard_normal(size=(nsim, n))
        sims = mu_hat + sig_hat * sims

        sims_sorted = np.sort(sims, axis=1)

        lo_q = (1.0 - band) / 2.0
        hi_q = 1.0 - lo_q
        lower = np.quantile(sims_sorted, lo_q, axis=0)
        upper = np.quantile(sims_sorted, hi_q, axis=0)

        # ---- Plot ----
        plt.fill_between(x, lower, upper, alpha=0.2, label=f"{int(band*100)}% envelope", color='C4')
        if has_line:
            plt.plot(*sup, linestyle="--", color="orange", label=lab)

        plt.scatter(x, y, s=12, color='C0')
        plt.grid(alpha=0.5, linestyle="--")
        plt.title("QQ Plot")
        plt.legend()
        plt.xlabel("Theoretical Quantiles (Normal)")
        plt.ylabel("Sample Quantiles")
        plt.show()

    # def qq(self, line: Literal["r", "q", "45"] | None = None):
    #     y = np.sort(self.resid)
    #     n = len(y)

    #     def std_norm_inv_cdf(p: float | list[float]):
    #         return np.float64(np.sqrt(2)*erfinv(2*p-1))
        
    #     def _qq_points(resid):
    #         y = np.sort(resid.astype(np.float64))
    #         n = len(y)
    #         p = (np.arange(1, n + 1) - 0.5) / n
    #         x = std_norm_inv_cdf(p)
    #         return x, y

    #     p = (np.arange(1, n+1) - 0.5) / n
    #     x = np.array(list(map(std_norm_inv_cdf, p)))
        
    #     sup, lab = None, None
    #     has_line = True
    #     match line:
    #         case "q":
    #             y1, y3 = np.percentile(y, [25, 75])
                
    #             x1 = std_norm_inv_cdf(.25)
    #             x3 = std_norm_inv_cdf(.75)

    #             m = (y3-y1)/(x3-x1)
    #             b = y1-m*x1

    #             sup = (x, m*x+b)
    #             lab = "Quartile Fit"
            
    #         case "r":
    #             m, b = np.polyfit(x, y, 1)
    #             sup = (x, m*x+b)
    #             lab = "Regression Line"
            
    #         case "45":
    #             mu = y.mean()  # approx 0
    #             sig = y.std(ddof=1) #unb. stdev
    #             sup = (x, mu+x*sig) # stardardized diagonal
    #             lab = r"$x=y$"
            
    #         case _:
    #             has_line = False

    #     plt.plot(*sup, label=lab, linestyle='--', color='orange') if has_line else None
    #     plt.scatter(x, y)
    #     plt.legend()
    #     plt.grid(alpha=0.5, linestyle='--')
    #     plt.show()


        


    def __repr__(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in asdict(self).items()])

# ================= QQ Helpers =================

def _std_norm_inv_cdf_vec(p: NDArray[np.float64]) -> NDArray[np.float64]:
    # uses erfinv; assume you already have it
    return np.array([np.float64(np.sqrt(2.0) * erfinv(2.0*p_i - 1.0)) for p_i in p])

def _qq_points(
    resid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return theoretical normal quantiles x and sorted sample quantiles y."""
    y = np.sort(resid.astype(np.float64))
    n = len(y)
    p = (np.arange(1, n + 1) - 0.5) / n
    x = _std_norm_inv_cdf_vec(p)
    return x, y