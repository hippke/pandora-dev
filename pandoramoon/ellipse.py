import numpy as np
from numpy import sqrt, pi, sin, cos, abs, tan, arctan
from numba import jit

from pandoramoon.trig import sincos_turns


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse(
    a, per, tau, Omega, i, time, x_bary, mass_ratio, b_bary
):
    """2D x-y Kepler solver without eccentricity"""

    O = Omega / 180 * pi
    i = i / 180 * pi
    xm = np.empty(len(time))
    ym = xm.copy()
    xp = xm.copy()
    yp = xm.copy()
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    k = pi * (time - tau * per) / per
    cos_Q = cos(2 * k)
    sin_Q = sin(2 * k)
    vector_x = cos(O) * cos_Q - sin(O) * sin_Q * cos(i)
    vector_y = sin(O) * cos_Q + cos(O) * sin_Q * cos(i)
    xm = +vector_x * a_moon + x_bary
    ym = +vector_y * a_moon + b_bary
    xp = -vector_x * a_planet + x_bary
    yp = -vector_y * a_planet + b_bary
    return xm, ym, xp, yp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_ecc(a, per, e, tau, Omega, w, i, time, mass_ratio, x_bary, b_bary):
    """2D x-y Kepler solver WITH eccentricity"""

    M = (2 * pi / per) * (time - (tau * per))
    M = M - (np.floor(M / (2 * pi)) * 2 * pi)
    flip = M > pi
    M = np.where(flip, 2 * pi - M, M)
    alpha = (3 * pi ** 2 + 1.6 * pi * (pi - abs(M)) / (1 + e)) / (pi ** 2 - 6)
    d = 3 * (1 - e) + alpha * e
    r1 = 3 * alpha * d * (d - 1 + e) * M + M ** 3
    q = 2 * alpha * d * (1 - e) - M ** 2
    w1 = (abs(r1) + sqrt(q ** 3 + r1 ** 2)) ** (2 / 3)
    n = (2 * r1 * w1 / (w1 ** 2 + w1 * q + q ** 2) + M) / d
    f0 = n - e * sin(n) - M
    f1 = 1 - e * cos(n)
    f2 = e * sin(n)
    g = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    h = -f0 / (f1 + 0.5 * g * f2 + (g ** 2) * (1 - f1) / 6)
    k = n - f0 / (f1 + 0.5 * h * f2 + h ** 2 * (1 - f1) / 6 + h ** 3 * (-f2) / 24)
    k = np.where(flip, 2 * pi - k, k)
    r = (1 - e * cos(k))

    wf = (w / 180 * pi) + (arctan(sqrt((1 + e) / (1 - e)) * tan(k / 2)) * 2)
    v = sin(wf) * cos((i / 180 * pi))
    O = Omega / 180 * pi
    vector_x = (cos(O) * cos(wf) - sin(O) * v) * r
    vector_y = (sin(O) * cos(wf) + cos(O) * v) * r

    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    xm = +vector_x * a_moon + x_bary
    ym = +vector_y * a_moon + b_bary
    xp = -vector_x * a_planet + x_bary
    yp = -vector_y * a_planet + b_bary
    return xm, ym, xp, yp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_z(a, per, tau, Omega, i, time, x_bary, mass_ratio, b_bary):
    """ellipse() fused into one loop that also returns the distances from the
    stellar centre: (xm, ym, xp, yp, zm, zp). No full-length temporaries
    (used by the culled likelihood path). The phase is reduced in turns and
    sin/cos come from trig.sincos_turns() (P7), so the loop vectorises; it
    agrees with ellipse() to the ulp of ellipse()'s phase argument 2 pi
    (t - tau per) / per (and is more accurate than it for large t)."""
    O = Omega / 180 * pi
    inc = i / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    cos_i = cos(inc)
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    n = len(time)
    xm = np.empty(n)
    ym = np.empty(n)
    xp = np.empty(n)
    yp = np.empty(n)
    zm = np.empty(n)
    zp = np.empty(n)
    t_ref = tau * per
    for j in range(n):
        # moon phase in turns; vectorisable sincos (P7) instead of libm
        sin_Q, cos_Q = sincos_turns((time[j] - t_ref) / per)
        vector_x = cos_O * cos_Q - sin_O * sin_Q * cos_i
        vector_y = sin_O * cos_Q + cos_O * sin_Q * cos_i
        xmj = vector_x * a_moon + x_bary[j]
        ymj = vector_y * a_moon + b_bary
        xpj = -vector_x * a_planet + x_bary[j]
        ypj = -vector_y * a_planet + b_bary
        xm[j] = xmj
        ym[j] = ymj
        xp[j] = xpj
        yp[j] = ypj
        zm[j] = sqrt(xmj ** 2 + ymj ** 2)
        zp[j] = sqrt(xpj ** 2 + ypj ** 2)
    return xm, ym, xp, yp, zm, zp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_ecc_z(a, per, e, tau, Omega, w, i, time, mass_ratio, x_bary, b_bary):
    """ellipse_ecc() fused into one loop that also returns the distances from
    the stellar centre: (xm, ym, xp, yp, zm, zp). Same formulas per sample,
    including the per-sample reflection M > pi (bug #2)."""
    w_rad = w / 180 * pi
    cos_i = cos(i / 180 * pi)
    O = Omega / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    ecc_factor = sqrt((1 + e) / (1 - e))
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    n = len(time)
    xm = np.empty(n)
    ym = np.empty(n)
    xp = np.empty(n)
    yp = np.empty(n)
    zm = np.empty(n)
    zp = np.empty(n)
    for j in range(n):
        M = (2 * pi / per) * (time[j] - (tau * per))
        M = M - (np.floor(M / (2 * pi)) * 2 * pi)
        flip = M > pi
        if flip:
            M = 2 * pi - M
        alpha = (3 * pi ** 2 + 1.6 * pi * (pi - abs(M)) / (1 + e)) / (pi ** 2 - 6)
        d = 3 * (1 - e) + alpha * e
        r1 = 3 * alpha * d * (d - 1 + e) * M + M ** 3
        q = 2 * alpha * d * (1 - e) - M ** 2
        w1 = (abs(r1) + sqrt(q ** 3 + r1 ** 2)) ** (2 / 3)
        nn = (2 * r1 * w1 / (w1 ** 2 + w1 * q + q ** 2) + M) / d
        f0 = nn - e * sin(nn) - M
        f1 = 1 - e * cos(nn)
        f2 = e * sin(nn)
        g = -f0 / (f1 - 0.5 * f0 * f2 / f1)
        h = -f0 / (f1 + 0.5 * g * f2 + (g ** 2) * (1 - f1) / 6)
        k = nn - f0 / (f1 + 0.5 * h * f2 + h ** 2 * (1 - f1) / 6 + h ** 3 * (-f2) / 24)
        if flip:
            k = 2 * pi - k
        r = (1 - e * cos(k))
        wf = w_rad + (arctan(ecc_factor * tan(k / 2)) * 2)
        v = sin(wf) * cos_i
        cos_wf = cos(wf)
        vector_x = (cos_O * cos_wf - sin_O * v) * r
        vector_y = (sin_O * cos_wf + cos_O * v) * r
        xmj = vector_x * a_moon + x_bary[j]
        ymj = vector_y * a_moon + b_bary
        xpj = -vector_x * a_planet + x_bary[j]
        ypj = -vector_y * a_planet + b_bary
        xm[j] = xmj
        ym[j] = ymj
        xp[j] = xpj
        yp[j] = ypj
        zm[j] = sqrt(xmj ** 2 + ymj ** 2)
        zp[j] = sqrt(xpj ** 2 + ypj ** 2)
    return xm, ym, xp, yp, zm, zp