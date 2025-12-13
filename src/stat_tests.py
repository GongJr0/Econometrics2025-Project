import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Literal
from .fit_data import StatsTest
from .tau_coefs import mackinnon_p
from .sw_coefs import swilk
from .error_functions import r2
from .StandardScaler import StandardScaler
from math import gamma, erf
from mpmath import gammainc, betainc
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

def T_CDF(stat: float64, df: int) -> float64:
    """Compute the CDF of the t-distribution at a given statistic value.

    Args:
        stat (float): The t-statistic
        df (int): Degrees of freedom.
    Returns:
        float: The CDF value at the given statistic.
    """
    stat = float(stat)
    df = float(df)
    a = df / 2.0
    b = 0.5
    x = df / (df + stat*stat)
    I = betainc(a, b, 0, x, regularized=True)  # regularized incomplete beta

    if stat >= 0:
        # upper tail of beta gives upper tail of t
        return 1.0 - 0.5 * I
    else:
        return 0.5 * I
    
    

def F_CDF(stat: float64, dfn: int, dfd: int) -> float64:
    """Compute the CDF of the F-distribution at a given statistic value.

    Args:
        stat (float): The F-statistic value.
        dfn (int): Degrees of freedom in the numerator.
        dfd (int): Degrees of freedom in the denominator.
    Returns:
        float: The CDF value at the given statistic.
    """
    
    x = (dfn * stat) / (dfn * stat + dfd)
    return float64(betainc(dfn / 2, dfd / 2, 0, x, regularized=True))

def F_TEST(stat: float64, dfn: int, dfd: int, alpha: float64 = 0.05) -> StatsTest:
    """Perform an F-test given the F-statistic and degrees of freedom.

    Args:
        stat (float): The F-statistic
        dfn (int): Degrees of freedom in the numerator.
        dfd (int): Degrees of freedom in the denominator.
        alpha (float): Significance level for the test.
    Returns:
        StatsTest: Result of the F-test.
    """
    pval = 1 - F_CDF(stat, dfn, dfd)
    return StatsTest(
        reject=pval < alpha,
        pval=pval,
        test_stat=float(stat),
        stat_name="F Test (F-Statistic)"
    )

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

    try:
        beta_hat, *_ = np.linalg.lstsq(Xreg, y, rcond=-1)
    except np.linalg.LinAlgError as e:
        return StatsTest(
            reject=False,
            pval=np.inf,
            test_stat=np.nan,
            stat_name="ADF Test (T-Statistic) [Non-Convergent input]"
        )
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

    ddof = X.shape[1]  # = k-1 because intercept is not inclded yet.

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


def ecdf(x):
    """Compute the empirical cumulative distribution function (ECDF) for a 1D array."""
    x = np.asarray(x, dtype=float64).ravel()
    n = len(x)
    x_s = np.sort(x)
    y = np.array([(i + 1) / n for i in range(n)], dtype=float64)
    return x_s, y


def ecdf_F(ecdf_x, x_i):
    """Evaluate the ECDF at a specific point x_i."""
    idx = np.searchsorted(ecdf_x[0], x_i, side='right') - 1
    if idx < 0:
        return 0.0
    else:
        return ecdf_x[1][idx]

def D_CRIT(alpha, n, m):
        """Compute the critical value for the KS test at significance level alpha."""
        return float64(np.sqrt(-0.5 * np.log(alpha / 2.0))) * float64(np.sqrt((n + m) / (n * m)))


def Standardized_KS(X, Y, alpha=0.05, pval_terms=101) -> StatsTest:
    """Compute the Kolmogorov-Smirnov test statistic between mean-variance standardized samples X and Y."""
    s = StandardScaler()
    X = s.fit_transform(X.reshape(-1, 1)).ravel().astype(float64)
    Y = s.fit_transform(Y.reshape(-1, 1)).ravel().astype(float64)

    ecdf_X = ecdf(X)
    ecdf_Y = ecdf(Y)

    def obj(x):
        return abs(ecdf_F(ecdf_X, x) - ecdf_F(ecdf_Y, x))

    def ks_pval(D, n, m=None, terms=pval_terms):
        if m is None:
            # one-sample scaling
            lam = np.sqrt(n) * D
        else:
            # two-sample scaling
            lam = np.sqrt(n * m / (n + m)) * D

        # series expansion
        s = 0.0
        for k in range(1, terms + 1):
            s += (-1) ** (k - 1) * np.exp(-2 * (k ** 2) * (lam ** 2))

        return max(0.0, min(1.0, 2 * s))

    # Evaluate D at all sample points of X and Y
    XY = np.concatenate((X, Y))
    D_vals = np.array([obj(xi) for xi in XY], dtype=float64)
    D_idx = int(np.argmax(D_vals))
    D = float64(D_vals[D_idx])

    n = len(X)
    m = len(Y)
    D_crit = D_CRIT(alpha, n, m)
    pval = ks_pval(D, n, m, terms=pval_terms)

    return StatsTest(
        reject=bool(D > D_crit),
        pval=float64(pval),
        test_stat=float64(D),
        stat_name="Standardized Kolmogorov-Smirnov Test (D Statistic)",
    )


def VectorKS2Samp(X, alpha=0.05, pval_terms=101) -> tuple[NDArray[float64], NDArray[bool], NDArray[StatsTest]]:
    """Compute the Kolmogorov-Smirnov test statistic for each dimension of the input 2D array X."""
    X = np.asarray(X, dtype=float64)
    n_features = X.shape[1]
    ks_stats = np.zeros((n_features, n_features), dtype=float64)
    ks_reject_map = np.zeros((n_features, n_features), dtype=bool)
    ks_tests = np.empty((n_features, n_features), dtype=object)

    for i in range(n_features):
        for j in range(n_features):
            ks_test = Standardized_KS(X[:, i], X[:, j], alpha=alpha, pval_terms=pval_terms)
            ks_stats[i, j] = ks_test.test_stat
            ks_reject_map[i, j] = bool(ks_test.reject)
            ks_tests[i, j] = ks_test

    return ks_stats, ks_reject_map, ks_tests


def KS2Sample(X, alpha=0.05, pval_terms=101, display_plot=False, varnames: list[str] | None = None):
    """Compute the Kolmogorov-Smirnov test statistic between all columns in X."""
    ks_stats, ks_reject, tests = VectorKS2Samp(X, alpha=alpha, pval_terms=pval_terms)

    if not display_plot:
        return tests

    # scoped imports
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        cols = varnames or X.columns.tolist()
    else:
        cols = varnames or [f"X{i+1}" for i in range(X.shape[1])]

    ks_reject_df = pd.DataFrame(ks_reject, index=cols, columns=cols)
    ks_stats_df = pd.DataFrame(ks_stats, index=cols, columns=cols)

    ks_pvals = [[tests[i, j].pval for j in range(len(cols))] for i in range(len(cols))]
    ks_pvals_df = pd.DataFrame(ks_pvals, index=cols, columns=cols)
    
    nr_template = lambda x, y: f"$\mathbf{{H_0}}$ \n $p={x:.4f}$ \n $C = {y:.4f}$"
    r_template = lambda x, y: f"$\mathbf{{H_1}}$\n $p={x:.4f}$ \n $C = {y:.4f}$"
    annot_arr = []
    for i in range(ks_reject_df.shape[0]):
        for j in range(ks_reject_df.shape[1]):
            pval_ij = ks_pvals_df.iloc[i, j]
            if ks_reject_df.iloc[i, j]:
                annot_arr.append(r_template(pval_ij, ks_stats_df.iloc[i, j]))
            else:
                annot_arr.append(nr_template(pval_ij, ks_stats_df.iloc[i, j]))

    fig = plt.figure(figsize=(12, 8))

    # KS Reject Matrix (green → red)
    sns.heatmap(
        ks_reject_df,
        cmap="RdYlGn_r",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot=np.array(annot_arr).reshape(ks_reject_df.shape),
        annot_kws={'fontsize': 14, "weight": "bold"},
        fmt="",
        vmin=0,
        vmax=1,
    )
    plt.title(f"KS Test Matrix (α = {alpha})", fontsize=18, weight="bold")
    plt.tick_params(axis="x", rotation=0, labelsize=14)
    plt.tick_params(axis="y", rotation=0, labelsize=14)
    return ks_stats_df, ks_reject_df, ks_pvals


def auto_b_len(n: int) -> int:
    return np.round(n**(1/3)).astype(np.int64)


def block_bootstrap(X, *, block_len: int | None = None, rng: np.random.Generator | None = None) -> NDArray[float64]:
    """Generate a block-bootstrap sample from the input 1D array X."""
    X = np.asarray(X, dtype=float64).ravel()
    n = len(X)
    
    b_len = block_len or auto_b_len(n)
    n_blocks = int(np.ceil(n / b_len))
    
    rng = rng or np.random.default_rng()
    block_sizes = np.array([b_len] * (n_blocks - 1) + [n - b_len * (n_blocks - 1)])
    
    blocks = []
    for size in block_sizes:
        start_idx = rng.integers(0, n - size + 1)
        block = np.arange(start_idx, start_idx + size, dtype=np.int64)
        blocks.append(block)
    return np.asarray(np.concatenate(blocks), dtype=np.int64)

def BootstrapKS2Samp(X, block_len: int | None = None, n_bootstrap: int = 1000, alpha: float = 0.05, display_plot: bool = False, varnames: list[str] | None = None):
    """Perform block-bootstrap KS tests on all pairs of columns in X."""
    s = StandardScaler()
    X = s.fit_transform(np.asarray(X, dtype=float64))
    n_features = X.shape[1]
    T = X.shape[0]
    
    ks_stats = np.zeros((n_features, n_features), dtype=float64)
    ks_reject = np.zeros((n_features, n_features), dtype=bool)
    ks_tests = np.empty((n_features, n_features), dtype=object)
    no_boot_crit = D_CRIT(alpha, T, T)
    for i in range(n_features):
        for j in range(n_features):
            Xi = X[:, i]
            Xj = X[:, j]
            
            pool = np.concatenate([Xi, Xj])
            T = pool.shape[0]
            
            crit = D_CRIT(alpha, len(Xi), len(Xj))
            D_boot = np.zeros(n_bootstrap, dtype=float64)
            obs_stat = Standardized_KS(Xi, Xj, alpha=alpha).test_stat
            for b in range(n_bootstrap):
                idx = block_bootstrap(pool, block_len=block_len)
                Xi_b = idx[:len(Xi)]
                Xj_b = idx[len(Xi):]
                
                Xi_b = pool[Xi_b]
                Xj_b = pool[Xj_b]
                
                ks_test_b = Standardized_KS(Xi_b, Xj_b, alpha=alpha)
                D_boot[b] = ks_test_b.test_stat
                
            D_pval_est = (np.sum(D_boot >= obs_stat) + 1) / (n_bootstrap + 1)
            ks_stats[i, j] = obs_stat
            ks_reject[i, j] = D_pval_est < alpha
            ks_tests[i, j] = StatsTest(
                reject=bool(D_pval_est < alpha),
                pval=float(D_pval_est),
                test_stat=float(obs_stat),
                stat_name=("Block Bootstrap KS Test (D Statistic)" 
                          "[ONLY USE THE BOOTSTRAP P-VAL ESTIMATE. D IS JUST THE MEAN OF BOOTSTRAP D'S]")
            ) 
    
    if display_plot:  
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            cols = varnames or X.columns.tolist()
        else:
            cols = varnames or [f"X{i+1}" for i in range(X.shape[1])]

        ks_reject_df = pd.DataFrame(ks_reject, index=cols, columns=cols)
        ks_stats_df = pd.DataFrame(ks_stats, index=cols, columns=cols)

        ks_pvals = [[ks_tests[i, j].pval for j in range(len(cols))] for i in range(len(cols))]
        ks_pvals_df = pd.DataFrame(ks_pvals, index=cols, columns=cols)

        fig = plt.figure(figsize=(12, 8))

        nr_template = lambda x, y: f"$\mathbf{{H_0}}$ \n $p={x:.4f}$ \n $C = {y:.4f}$"
        r_template = lambda x, y: f"$\mathbf{{H_1}}$\n $p={x:.4f}$ \n $C = {y:.4f}$"
        annot_arr = []
        for i in range(ks_reject_df.shape[0]):
            for j in range(ks_reject_df.shape[1]):
                pval_ij = ks_pvals_df.iloc[i, j]
                if ks_reject_df.iloc[i, j]:
                    annot_arr.append(r_template(pval_ij, ks_stats_df.iloc[i, j]))
                else:
                    annot_arr.append(nr_template(pval_ij, ks_stats_df.iloc[i, j]))
        # Old annot: np.where(ks_reject_df.values == 1, r"Reject $\mathbf{H_0}$", "Fail to Reject")

        # KS Reject Matrix (green → red)
        sns.heatmap(
            ks_reject_df,
            cmap="RdYlGn_r",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            annot=np.array(annot_arr).reshape(ks_reject_df.shape),
            fmt="",
            annot_kws={"weight": "bold", "fontsize": 14},
            vmin=0,
            vmax=1,
        )
        plt.title(f"Bootstrapped KS Test Matrix (α = {alpha})", fontsize=14, weight="bold")
        plt.tick_params(axis="x", rotation=0, labelsize=14)
        plt.tick_params(axis="y", rotation=0, labelsize=14)
    
    return ks_stats, ks_pvals, ks_reject, ks_tests
    
def VIF(X: NDArray[float64]) -> NDArray[float64]:
    """Compute the Variance Inflation Factor (VIF) for each column in the input 2D array X."""
    X = np.asarray(X, dtype=float64)
    n_features = X.shape[1]
    vif_values = np.zeros(n_features, dtype=float64)

    for i in range(n_features):
        y = X[:, i]
        est = np.delete(X, i, axis=1)
        X_vif = np.column_stack([np.ones(est.shape[0]), est])
        XT_X = X_vif.T @ X_vif
        beta = np.linalg.inv(XT_X) @ X_vif.T @ y
        fitted = X_vif @ beta

        r2_val = r2(y, fitted)
        vif_values[i] = 1.0 / (1.0 - r2_val)
    
    return vif_values

def DW(eps: NDArray[float64], X:NDArray[float64], idx_ar: int = 0, alpha: float = 0.05) -> StatsTest:
    n = len(eps)
    k = X.shape[1]
    resid_var = (eps.T @ eps) / (n - k)
    M = np.linalg.inv(X.T @ X)
    var_b = resid_var * M
    se_b = np.sqrt(np.diag(var_b))

    print("SE: ", se_b)

    d_num = (np.sum((eps[1:] - eps[:-1])**2))
    d_denom = np.sum(eps**2)

    d = d_num / d_denom

    print("D: ", d)

    h = (1 - d/2) * np.sqrt(n/(1 - n*se_b[idx_ar]**2))

    print("H: ", h)

    def norm_cdf(x):
        return (1/2) * (1 + erf(x/np.sqrt(2))) * 2  # two-tailed
    
    pval = norm_cdf(-abs(h))
    return StatsTest(
        reject=pval<alpha,
        pval=float(pval),
        test_stat=float(h),
        stat_name="Durbin-Watson H Test (Z-Statistic)"
    )

def BG(eps: NDArray[float64], p: int = 1, alpha: float = 0.05) -> StatsTest:
    T = eps.shape[0]
    
    y = eps[p:]
    X = np.zeros((T-p, p), dtype=float64)
    for i in range(p):
        X[:, i] = eps[p - i - 1: T - i - 1]
    
    XT_X = X.T @ X
    beta = np.linalg.inv(XT_X) @ X.T @ y
    fitted = X @ beta

    r2_val = r2(y, fitted)
    bg_stat = (T-p) * r2_val

    pval = gammainc(p/2, 0, bg_stat/2)/gamma(p/2)  # Chi^2(p) CDF

    return StatsTest(
        reject=pval<alpha,
        pval=float(pval),
        test_stat=float(bg_stat),
        stat_name=f"BG({p}) Test (Chi^2 Statistic)"
    )
    
def RESET(y: NDArray[float64], y_hat: NDArray[float64], X: NDArray[float64], q: int = 2, alpha: float64 = 0.05) -> StatsTest:
    
    n = y.shape[0]
    k = X.shape[1]

    unrestricted_X = np.column_stack([X, *[y_hat**(i+2) for i in range(q)]])
    uXT_X = unrestricted_X.T @ unrestricted_X
    u_beta = np.linalg.inv(uXT_X) @ unrestricted_X.T @ y
    u_fitted = unrestricted_X @ u_beta
    r2_ur = r2(y, u_fitted)
    
    res_fitted = X @ (np.linalg.inv(X.T @ X) @ X.T @ y)
    r2_r = r2(y, res_fitted)
    
    F_num = (r2_ur - r2_r) / q
    F_denom = (1 - r2_ur) / (n-k-q)
    
    F_stat = F_num / F_denom
    
    pval = 1 - F_CDF(F_stat, q, n - k - q)
    return StatsTest(
        reject=pval<alpha,
        pval=float(pval),
        test_stat=float(F_stat),
        stat_name=f"RESET Test (F-Statistic)"
    )
    
def HC(y: NDArray[float64], X:NDArray[float64], alpha: float = 0.05) -> StatsTest:
    y = np.asarray(y, dtype=float64).ravel()
    X = np.asarray(X, dtype=float64)
    n, k = X.shape

    w_list: list[float] = []

    for t in range(k, n):
        X_t1 = X[:t, :]
        y_t1 = y[:t]

        XtX = X_t1.T @ X_t1
        beta_t1 = np.linalg.solve(XtX, X_t1.T @ y_t1)

        x_t = X[t, :]
        y_hat_t = float(x_t @ beta_t1)
        e_t = float(y[t] - y_hat_t)

        XtX_inv = np.linalg.inv(XtX)
        d_t = float(np.sqrt(1.0 + x_t @ XtX_inv @ x_t))

        w_list.append(e_t / d_t)

    w = np.asarray(w_list, dtype=float64)
    m = w.shape[0]

    w_bar = float(w.mean())
    s2 = float(np.sum((w - w_bar)**2) / (m - 1))

    se_mean = float(np.sqrt(s2 / m))
    t_stat = w_bar / se_mean
    df = m - 1

    pval = 2.0 * (1.0 - T_CDF(abs(t_stat), df))

    return StatsTest(
        reject=bool(pval < alpha),
        pval=float(pval),
        test_stat=float(t_stat),
        stat_name="Harvey-Collier Test (t-statistic)",
    )
    
    