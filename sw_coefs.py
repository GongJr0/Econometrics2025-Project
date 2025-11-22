from numpy import array, float64
import numpy as np

# Ploynomial Coefficients Copied from SciPy

c1 = array(
    [0., 0.221157, -0.147981, -0.207119e1, 0.4434685e1, -0.2706056e1],
    dtype=float64
    )


c2 = array(
    [0., 0.42981e-1, -0.293762, -0.1752461e1, 0.5682633e1, -0.3582633e1],
    dtype=float64
    )

c3 = array(
    [0.5440, -0.39978, 0.25054e-1, -0.6714e-3],
    dtype=float64
    )

c4 = array(
    [0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2],
    dtype=float64
)

c5 = array(
    [-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2],
    dtype=float64
)

c6 = array(
    [-0.4803, -0.82676e-1, 0.30302e-2],
    dtype=float64
)

c7 = array(
    [0.164, 0.533],
    dtype=float64
)

c8 = array(
    [0.1736, 0.315],
    dtype=float64
)

c9 = array(
    [0.256, -0.635e-2],
    dtype=float64
)

g = array(
    [-0.2273e1, 0.459],
    dtype=float64
)

def _ppnd(p):
    A0 = 2.50662823884
    A1 = -18.61500062529
    A2 = 41.39119773534
    A3 = -25.44106049637
    B1 = -8.47351093090
    B2 = 23.08336743743
    B3 = -21.06224101826
    B4 = 3.13082909833
    C0 = -2.78718931138
    C1 = -2.29796479134
    C2 = 4.85014127135
    C3 = 2.32121276858
    D1 = 3.54388924762
    D2 = 1.63706781897

    split = 0.42
    q = p - 0.5
    if abs(q) <= split:
        r = q * q
        temp = q*(((A3*r + A2)*r + A1) * r + A0)
        temp = temp / ((((B4*r + B3)*r + B2)*r + B1)*r + 1.)
        return temp  #, 0

    r = p
    if q > 0:
        r = 1 - p
    if r > 0:
        r = np.sqrt(-np.log(r))
    else:
        return 0.  #, 1

    temp = (((C3*r + C2)*r + C1)*r + C0)
    temp /= (D2*r + D1)*r + 1.
    return (-temp if q < 0 else temp)  #, 0

def _poly(c, nord, x):
    """
    Helper function for swilk function that evaluates polynomials.
    For some reason, the coefficients are given as

        [c0, cn, cn-1, ..., c2, c1]

    hence the backwards loop.
    """
    res = c[0]
    if nord == 1:
        return res

    p = x*c[nord-1]
    if nord == 2:
        return res + p

    for ind in range(nord-2, 0, -1):
        p = (p + c[ind])*x
    res += p
    return res

def _alnorm(x, upper):
    """
    Helper function for swilk.

    Evaluates the tail area of the standardized normal curve from x to inf
    if upper is True or from -inf to x if upper is False

    Modification has been done to the Fortran version in November 2001 with the
    following note;

        MODIFY UTZERO.  ALTHOUGH NOT NECESSARY
        WHEN USING ALNORM FOR SIMPLY COMPUTING PERCENT POINTS,
        EXTENDING RANGE IS HELPFUL FOR USE WITH FUNCTIONS THAT
        USE ALNORM IN INTERMEDIATE COMPUTATIONS.

    The change is shown below as a commented utzero definition
    """
    ltone = 7.
    utzero = 38.
    con = 1.28

    A1 = 0.398942280444
    A2 = 0.399903438504
    A3 = 5.75885480458
    A4 = 29.8213557808
    A5 = 2.62433121679
    A6 = 48.6959930692
    A7 = 5.92885724438
    B1 = 0.398942280385
    B2 = 3.8052e-8
    B3 = 1.00000615302
    B4 = 3.98064794e-4
    B5 = 1.98615381364
    B6 = 0.151679116635
    B7 = 5.29330324926
    B8 = 4.8385912808
    B9 = 15.1508972451
    B10 = 0.742380924027
    B11 = 30.789933034
    B12 = 3.99019417011
    z = x
    if not (z > 0):  # negative of the condition to catch NaNs
        upper = False
        z = -z
    if not ((z <= ltone) or (upper and z <= utzero)):
        return 0. if upper else 1.
    y = 0.5 * z * z
    if z <= con:
        temp = 0.5 - z*(A1 - A2*y/(y + A3 - A4/(y + A5 + A6/(y + A7))))
    else:
        temp = B1 * np.exp(-y)/(z - B2 + B3/(z + B4 + B5/(z - B6 + B7/
                             (z + B8 - B9/(z + B10 + B11/(z + B12))))))

    return temp if upper else (1-temp)

def swilk(x, a, init, n1=-1):
    Z90 = 0.12816e1
    Z95 = 0.16449e1
    Z99 = 0.23263e1
    ZM = 0.17509e1
    ZSS = 0.56268
    BF1 = 0.8378
    XX90 = 0.556
    XX95 = 0.622

    SQRTH = np.sqrt(2)/2
    PI6 = 6/np.pi
    SMALL=1e-19

    n=len(x)
    n2 = len(a)
    upper = True

    if n1 < 0:
        n1 = n
    nn2 = n // 2
    if nn2 < n2:
        return 1., 1., 3
    if n < 3:
        return 1., 1., 1
    w = 1.
    pw = 1.
    an = n

    if not init:
        if n == 3:
            a[0] = SQRTH
        else:
            an25 = an + 0.25
            summ2 = 0.
            for ind1 in range(n2):
                temp = _ppnd((ind1+1-0.375)/an25)
                a[ind1] = temp
                summ2 += temp**2

            summ2 *= 2.
            ssumm2 = np.sqrt(summ2)
            rsn = 1 / np.sqrt(an)
            A1 = _poly(c1, 6, rsn) - (a[0]/ssumm2)
            if n > 5:
                i1 = 2
                A2 = -a[1]/ssumm2 + _poly(c2, 6, rsn)
                fac = np.sqrt((summ2 - (2 * a[0]**2) - 2 * a[1]**2) /
                           (1 - (2 * A1**2) - 2 * A2**2))
                a[1] = A2
            else:
                i1 = 1
                fac = np.sqrt((summ2 - 2 * a[0]**2)/(1 - 2 * A1**2))

            a[0] = A1
            for ind1 in range(i1, nn2):
                a[ind1] *= -1./fac
        init = True

    if n1 < 3:
        return w, pw, 1
    ncens = n - n1

    if ncens < 0 or ((ncens > 0) and (n < 20)):
        return w, pw, 4
    delta = ncens / an
    if delta > 0.8:
        return w, pw, 5
    
    RANGE = x[n1-1] - x[0]
    if RANGE < SMALL:
        return w, pw, 6

    XX = x[0] / RANGE
    SX = XX
    SA = -a[0]
    ind2 = n - 2
    for ind1 in range(1, n1):
        XI = x[ind1] / RANGE
        SX += XI
        if ind1 != ind2:
            SA += (-1 if ind1 < ind2 else 1)*a[min(ind1, ind2)]
        XX = XI
        ind2 -= 1

    ifault = 0
    if n > 5000:
        ifault = 2

    SA /= n1
    SX /= n1
    SSA, SSX, SAX = 0., 0., 0.
    ind2 = n - 1
    for ind1 in range(n1):
        if ind1 != ind2:
            ASA = (-1 if ind1 < ind2 else 1)*a[min(ind1, ind2)] - SA
        else:
            ASA = -SA

        XSX = x[ind1]/RANGE - SX
        SSA += ASA * ASA
        SSX += XSX * XSX
        SAX += ASA * XSX
        ind2 -= 1

    SSASSX = np.sqrt(SSA * SSX)
    w1 = (SSASSX - SAX) * (SSASSX + SAX)/(SSA * SSX)
    w = 1 - w1

    # Calculate significance level for W (exact for N=3)
    if n == 3:
        # Original Fortran code computation was below
        #
        # pw = PI6 * (asin(sqrt(w)) - PI_OVER_3)
        #
        # However this can return negative p-values for N==3;
        # see gh-18322 and also 32-bit Linux systems.
        # Thus, a potential improvement: precision for small p-values
        # Theoretically w >= 0.75, hence clamping the value
        if w < 0.75:
            return 0.75, 0., ifault
        else:
            pw = 1. - PI6 * np.acos(np.sqrt(w))
            return w, pw, ifault

    y = np.log(w1)
    XX = np.log(an)
    if n <= 11:
        gamma = _poly(g, 2, an)
        if y >= gamma:
            return w, SMALL, ifault
        y = -np.log(gamma - y)
        m = _poly(c3, 4, an)
        s = np.exp(_poly(c4, 4, an))
    else:
        m = _poly(c5, 4, XX)
        s = np.exp(_poly(c6, 3, XX))

    if ncens > 0:
        ld = -np.log(delta)
        bf = 1 + XX*BF1
        Z90F = Z90 + bf * _poly(c7, 2, XX90 ** XX)**ld
        Z95F = Z95 + bf * _poly(c8, 2, XX95 ** XX)**ld
        Z99F = Z99 + bf * _poly(c9, 2, XX)**ld
        ZFM = (Z90F + Z95F + Z99F)/3.
        ZSD = (Z90*(Z90F-ZFM)+Z95*(Z95F-ZFM)+Z99*(Z99F-ZFM))/ZSS
        ZBAR = ZFM - ZSD * ZM
        m += ZBAR * s
        s *= ZSD

    pw = _alnorm((y-m)/s, upper)

    return w, pw, ifault