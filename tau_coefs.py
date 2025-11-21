from numpy import inf, array, asarray, float64, sqrt, polyval
from math import erf

from typing import Literal

# All code regarding tau coefficients was directly copied from statsmodels/tsa/adfvalues.py
# 
# Below are the coefficients are used in response-surface appriximations (MacKinnon, 1994) 
# for the Dickey-Fuller distribution critical values and p-values.
# The coefficients below are the second iteration of coefs released in 2010 as an update to the original 1994 paper facilitating ADF tests.

# ========= Interpretation of Arrays =========
# the first axis is N -1
# the second axis is 1 %, 5 %, 10 %
# the last axis is the coefficients
# "nc", "c", "ct", and "ctt" indicate the presence/absence and nature of trend-terms in the regression used.
# ==================================================

__all__ = ["mackinnon_p"]


# ========= Beginning of Copied Code =========
tau_star_nc = [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77]
tau_min_nc = [-19.04, -19.62, -21.21, -23.25, -21.63, -25.74]
tau_max_nc = [inf, 1.51, 0.86, 0.88, 1.05, 1.24]
tau_star_c = [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93]
tau_min_c = [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27]
tau_max_c = [2.74, 0.92, 0.55, 0.61, 0.79, 1]
tau_star_ct = [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36]
tau_min_ct = [-16.18, -21.15, -25.37, -26.63, -26.53, -26.18]
tau_max_ct = [0.7, 0.63, 0.71, 0.93, 1.19, 1.42]
tau_star_ctt = [-3.21, -3.51, -3.81, -3.83, -4.12, -4.63]
tau_min_ctt = [-17.17, -21.1, -24.33, -24.03, -24.33, -28.22]
tau_max_ctt = [0.54, 0.79, 1.08, 1.43, 3.49, 1.92]

_tau_maxs = {
    "n": tau_max_nc,
    "c": tau_max_c,
    "ct": tau_max_ct,
    "ctt": tau_max_ctt,
}
_tau_mins = {
    "n": tau_min_nc,
    "c": tau_min_c,
    "ct": tau_min_ct,
    "ctt": tau_min_ctt,
}
_tau_stars = {
    "n": tau_star_nc,
    "c": tau_star_c,
    "ct": tau_star_ct,
    "ctt": tau_star_ctt,
}


small_scaling = array([1, 1, 1e-2])
tau_nc_smallp = [
    [0.6344, 1.2378, 3.2496],
    [1.9129, 1.3857, 3.5322],
    [2.7648, 1.4502, 3.4186],
    [3.4336, 1.4835, 3.19],
    [4.0999, 1.5533, 3.59],
    [4.5388, 1.5344, 2.9807]]
tau_nc_smallp = asarray(tau_nc_smallp)*small_scaling

tau_c_smallp = [
    [2.1659, 1.4412, 3.8269],
    [2.92, 1.5012, 3.9796],
    [3.4699, 1.4856, 3.164],
    [3.9673, 1.4777, 2.6315],
    [4.5509, 1.5338, 2.9545],
    [5.1399, 1.6036, 3.4445]]
tau_c_smallp = asarray(tau_c_smallp)*small_scaling

tau_ct_smallp = [
    [3.2512, 1.6047, 4.9588],
    [3.6646, 1.5419, 3.6448],
    [4.0983, 1.5173, 2.9898],
    [4.5844, 1.5338, 2.8796],
    [5.0722, 1.5634, 2.9472],
    [5.53, 1.5914, 3.0392]]
tau_ct_smallp = asarray(tau_ct_smallp)*small_scaling

tau_ctt_smallp = [
    [4.0003, 1.658, 4.8288],
    [4.3534, 1.6016, 3.7947],
    [4.7343, 1.5768, 3.2396],
    [5.214, 1.6077, 3.3449],
    [5.6481, 1.6274, 3.3455],
    [5.9296, 1.5929, 2.8223]]
tau_ctt_smallp = asarray(tau_ctt_smallp)*small_scaling

_tau_smallps = {
    "n": tau_nc_smallp,
    "c": tau_c_smallp,
    "ct": tau_ct_smallp,
    "ctt": tau_ctt_smallp,
}


large_scaling = array([1, 1e-1, 1e-1, 1e-2])
tau_nc_largep = [
    [0.4797, 9.3557, -0.6999, 3.3066],
    [1.5578, 8.558, -2.083, -3.3549],
    [2.2268, 6.8093, -3.2362, -5.4448],
    [2.7654, 6.4502, -3.0811, -4.4946],
    [3.2684, 6.8051, -2.6778, -3.4972],
    [3.7268, 7.167, -2.3648, -2.8288]]
tau_nc_largep = asarray(tau_nc_largep)*large_scaling

tau_c_largep = [
    [1.7339, 9.3202, -1.2745, -1.0368],
    [2.1945, 6.4695, -2.9198, -4.2377],
    [2.5893, 4.5168, -3.6529, -5.0074],
    [3.0387, 4.5452, -3.3666, -4.1921],
    [3.5049, 5.2098, -2.9158, -3.3468],
    [3.9489, 5.8933, -2.5359, -2.721]]
tau_c_largep = asarray(tau_c_largep)*large_scaling

tau_ct_largep = [
    [2.5261, 6.1654, -3.7956, -6.0285],
    [2.85, 5.272, -3.6622, -5.1695],
    [3.221, 5.255, -3.2685, -4.1501],
    [3.652, 5.9758, -2.7483, -3.2081],
    [4.0712, 6.6428, -2.3464, -2.546],
    [4.4735, 7.1757, -2.0681, -2.1196]]
tau_ct_largep = asarray(tau_ct_largep)*large_scaling

tau_ctt_largep = [
    [3.0778, 4.9529, -4.1477, -5.9359],
    [3.4713, 5.967, -3.2507, -4.2286],
    [3.8637, 6.7852, -2.6286, -3.1381],
    [4.2736, 7.6199, -2.1534, -2.4026],
    [4.6679, 8.2618, -1.822, -1.9147],
    [5.0009, 8.3735, -1.6994, -1.6928]]
tau_ctt_largep = asarray(tau_ctt_largep)*large_scaling

_tau_largeps = {
    "n": tau_nc_largep,
    "c": tau_c_largep,
    "ct": tau_ct_largep,
    "ctt": tau_ctt_largep,
}


# NOTE: The Z-statistic is used when lags are included to account for
#  serial correlation in the error term

z_star_nc = [-2.9, -8.7, -14.8, -20.9, -25.7, -30.5]
z_star_c = [-8.9, -14.3, -19.5, -25.1, -29.6, -34.4]
z_star_ct = [-15.0, -19.6, -25.3, -29.6, -31.8, -38.4]
z_star_ctt = [-20.7, -25.3, -29.9, -34.4, -38.5, -44.2]


# These are Table 5 from MacKinnon (1994)
# small p is defined as p in .005 to .150 ie p = .005 up to z_star
# Z* is the largest value for which it is appropriate to use these
# approximations
# the left tail approximation is
# p = norm.cdf(d_0 + d_1*log(abs(z)) + d_2*log(abs(z))**2 + d_3*log(abs(z))**3)
# there is no Z-min, ie., it is well-behaved in the left tail

z_nc_smallp = array([
    [.0342, -.6376, 0, -.03872],
    [1.3426, -.7680, 0, -.04104],
    [3.8607, -2.4159, .51293, -.09835],
    [6.1072, -3.7250, .85887, -.13102],
    [7.7800, -4.4579, 1.00056, -.14014],
    [4.0253, -.8815, 0, -.04887]])

z_c_smallp = array([
    [2.2142, -1.7863, .32828, -.07727],
    [1.1662, .1814, -.36707, 0],
    [6.6584, -4.3486, 1.04705, -.15011],
    [3.3249, -.8456, 0, -.04818],
    [4.0356, -.9306, 0, -.04776],
    [13.9959, -8.4314, 1.97411, -.22234]])

z_ct_smallp = array([
    [4.6476, -2.8932, 0.5832, -0.0999],
    [7.2453, -4.7021, 1.127, -.15665],
    [3.4893, -0.8914, 0, -.04755],
    [1.6604, 1.0375, -0.53377, 0],
    [2.006, 1.1197, -0.55315, 0],
    [11.1626, -5.6858, 1.21479, -.15428]])

z_ctt_smallp = array([
    [3.6739, -1.1549, 0, -0.03947],
    [3.9783, -1.0619, 0, -0.04394],
    [2.0062, 0.8907, -0.51708, 0],
    [4.9218, -1.0663, 0, -0.04691],
    [5.1433, -0.9877, 0, -0.04993],
    [23.6812, -14.6485, 3.42909, -.33794]])
# These are Table 6 from MacKinnon (1994).
# These are well-behaved in the right tail.
# the approximation function is
# p = norm.cdf(d_0 + d_1 * z + d_2*z**2 + d_3*z**3 + d_4*z**4)
z_large_scaling = array([1, 1e-1, 1e-2, 1e-3, 1e-5])
z_nc_largep = array([
    [0.4927, 6.906, 13.2331, 12.099, 0],
    [1.5167, 4.6859, 4.2401, 2.7939, 7.9601],
    [2.2347, 3.9465, 2.2406, 0.8746, 1.4239],
    [2.8239, 3.6265, 1.6738, 0.5408, 0.7449],
    [3.3174, 3.3492, 1.2792, 0.3416, 0.3894],
    [3.729, 3.0611, 0.9579, 0.2087, 0.1943]])
z_nc_largep *= z_large_scaling

z_c_largep = array([
    [1.717, 5.5243, 4.3463, 1.6671, 0],
    [2.2394, 4.2377, 2.432, 0.9241, 0.4364],
    [2.743, 3.626, 1.5703, 0.4612, 0.567],
    [3.228, 3.3399, 1.2319, 0.3162, 0.3482],
    [3.6583, 3.0934, 0.9681, 0.2111, 0.1979],
    [4.0379, 2.8735, 0.7694, 0.1433, 0.1146]])
z_c_largep *= z_large_scaling

z_ct_largep = array([
    [2.7117, 4.5731, 2.2868, 0.6362, 0.5],
    [3.0972, 4.0873, 1.8982, 0.5796, 0.7384],
    [3.4594, 3.6326, 1.4284, 0.3813, 0.4325],
    [3.806, 3.2634, 1.0689, 0.2402, 0.2304],
    [4.1402, 2.9867, 0.8323, 0.16, 0.1315],
    [4.4497, 2.7534, 0.6582, 0.1089, 0.0773]])
z_ct_largep *= z_large_scaling

z_ctt_largep = array([
    [3.4671, 4.3476, 1.9231, 0.5381, 0.6216],
    [3.7827, 3.9421, 1.5699, 0.4093, 0.4485],
    [4.052, 3.4947, 1.1772, 0.2642, 0.2502],
    [4.3311, 3.1625, 0.9126, 0.1775, 0.1462],
    [4.594, 2.8739, 0.707, 0.1181, 0.0838],
    [4.8479, 2.6447, 0.5647, 0.0827, 0.0518]])
z_ctt_largep *= z_large_scaling

# ========= End of Copied Code =========


def mackinnon_p(stat: float64, trend: Literal["n", "c", "ct", "ctt"] = "c") -> float64:
    # N=1 for all ADF applications

    _MAX_STAT = _tau_maxs[trend][0]
    _MIN_STAT = _tau_mins[trend][0]

    _STAR_STAT = _tau_stars[trend][0]


    # Bounds check
    if stat > _MAX_STAT:
        return 1.0
    elif stat < _MIN_STAT:
        return 0.0

    # Get p region (large/small)
    if stat <= _STAR_STAT:
        coef = _tau_smallps[trend][0]
    else:
        coef = _tau_largeps[trend][0]
    
    norm_x = polyval(coef[::-1], stat)
    return (1/2) * (1+erf(norm_x/sqrt(2)))  # Standard Normal CDF
