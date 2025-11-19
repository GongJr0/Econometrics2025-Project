import numpy as np


# ================= Error Functions =================
def r2(y, y_hat) -> float:
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum((y - y_hat) ** 2)

    return 1 - (rss / tss)


def r2_adj(y, y_hat, n_x) -> float:
    n = len(y)

    num = (1 - r2(y, y_hat)) * (n - 1)
    denom = n - n_x - 1

    return 1 - (num / denom)


def mse(y, y_hat) -> float:
    n = len(y)
    rss = np.sum((y - y_hat) ** 2)

    return rss / n


def rmse(y, y_hat) -> float:
    return np.sqrt(mse(y, y_hat))


def mape(y, y_hat) -> float:
    n = len(y)
    p_err = np.sum(np.abs((y - y_hat) / y))
    return p_err / n
