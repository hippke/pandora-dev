import numpy as np
from numpy import sqrt, pi, arccos, abs, log, ceil
from numba import jit

from pandoramoon.trig import acos_poly


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def create_occult_cache(u1, u2, dim):
    """2D Cache of size dim*dim with Mandel-Agol occultation values
    as a function of quadratic limb darkening values (u1, u2), k (=radius ratio)"""
    z_max = 1.1
    k_min = 0.001
    k_max = 0.1
    zs = np.linspace(0, z_max, dim)
    ks = np.linspace(k_min, k_max, dim)
    fs = np.empty((dim, dim), dtype="float32")
    for count, k in enumerate(ks):
        fs[count] = occult(zs, k, u1, u2)
    return (fs, ks, zs)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def read_occult_cache(zs_target, k, cache):
    """Read nearest neighbors from 2D cache and perform bilinear interpolation"""
    fs, ks, zs = cache
    flux = np.ones(len(zs_target))
    idx_k = int(np.ceil((k - ks[0]) / (ks[1] - ks[0])))
    if idx_k < 1:
        idx_k = 1
    if idx_k > len(ks) - 1:
        idx_k = len(ks) - 1
    ratio_k = (ks[idx_k] - k) / (ks[idx_k] - ks[idx_k - 1])
    curve = fs[idx_k - 1] * ratio_k + fs[idx_k] * (1 - ratio_k)
    for idx in range(len(zs_target)):
        if zs_target[idx] == 0:
            idx_z = 1
        else:
            idx_z = int(np.ceil(zs_target[idx] / zs[1]))
        if zs_target[idx] >= 1.1:
            continue

        ratio_z = (zs[idx_z] - zs_target[idx]) / (zs[idx_z] - zs[idx_z - 1])
        res = curve[idx_z - 1] * ratio_z + curve[idx_z] * (1 - ratio_z)
        if res > 1:
            res = 1
        if res < 0:
            res = 0
        flux[idx] = res

        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if flux[idx] > 1:
            flux[idx] = 1
    return flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def cci(r1, r2, d):
    """Circle-Circle-Intersect to calculate the area of asymmetric "lens"
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    if r1 < d - r2:
        return 0
    elif r1 >= d + r2:
        return pi * r2**2
    elif d - r2 <= -r1:
        return pi * r1**2
    else:
        return (
            r2**2 * arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            + r1**2 * arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            - 0.5 * sqrt((-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1))
        )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_small(zs, k, u1, u2):
    """Small body approximation by Mandel-Agol. Adequate for k<~0.01"""
    i = zs.size
    f = np.ones(i)
    b = abs(zs)
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    s_inv = 1 / s
    for j in range(i):
        if b[j] < 1 + k:
            m = sqrt(1 - min(b[j] ** 2, 1))
            limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
            area = cci(1, k, b[j])
            f[j] = (s - limb_darkening * area) * s_inv

            # Some combinations of (u1,u2) cause flux>1 which is unphysical
            if f[j] > 1:
                f[j] = 1
    return f


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_small_single_value(z, k, u1, u2):
    """Small body approximation by Mandel-Agol. Adequate for k<~0.01
    This version does not calculate an entire array, but just one value.
    (Until P3 it was used by occult_hybrid's small-body interpolation zone,
    bug #28; kept for API compatibility.)"""
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    m = sqrt(1 - min(z**2, 1))
    limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
    area = cci(1, k, z)
    flux = (s - limb_darkening * area) * (1 / s)

    # Some combinations of (u1,u2) cause flux>1 which is unphysical
    if flux > 1:
        flux = 1
    return flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind
    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353
    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""
    HALF_PI = 0.5 * pi
    kc = sqrt(1 - k**2)
    e = kc
    p = sqrt(n + 1)
    m0 = 1
    c = 1
    d = 1 / p
    for nit in range(1000):
        f = c
        c = d / p + c
        g = e / p
        d = 2 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(1 - kc / g) > 1e-8:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return HALF_PI * (c * m0 + d) / (m0 * (m0 + p))
    raise ValueError("ellpicb did not converge")


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ell_combo(q, n, A, B, C):
    """A K(q) + B E(q) + C Pi(n, q) in one Bulirsch iteration (P3a).

    q is the modulus (K(q) = ellk(q), E(q) = ellec(q) up to their ~2e-8
    polynomial error), Pi(n, q) = ellpicb(n, q) = Bulirsch cel(kc, n+1, 1, 1).
    All three share the AGM sequence (kc, m0, e) of Bulirsch's cel
    algorithm (Bulirsch 1969, Numer. Math. 13, 305; as in ellpicb).
    K and E: cel(kc, 1, A + B, A + B kc^2); for p = 1 the cel p-sequence
    equals m0, so this pair needs no p update. Pi: cel(kc, n+1, C, C).
    Exact to ~1e-16 relative (quadratic convergence after the 1e-8 test),
    instead of ellk/ellec's ~2e-8."""
    p = sqrt(n + 1)
    return ell_combo_kc(sqrt(1 - q * q), p, A, B, C, C / p)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ell_combo_kc(kc, p, A, B, C, Cp):
    """ell_combo() with its operands given directly (P15): the complementary
    modulus kc = sqrt(1 - q^2) > 0, p = sqrt(n + 1) > 0 and Cp = C / p.
    Lets the caller form kc, p and C / p from (z, k) in closed form, without
    the cancellation of 1 - q^2 near q = 1 (Agol, Luger & Foreman-Mackey
    2020, eq. 41)."""
    a1 = A + B
    b1 = A + B * kc * kc
    a2 = C
    b2 = Cp
    e = kc
    m0 = 1.0
    for nit in range(1000):
        # K/E pair (its p equals m0)
        f1 = a1
        a1 = a1 + b1 / m0
        b1 = 2 * (b1 + f1 * kc)
        # Pi pair (as ellpicb)
        inv_p = 1 / p
        f2 = a2
        a2 = a2 + b2 * inv_p
        g = e * inv_p
        b2 = 2 * (b2 + f2 * g)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(g - kc) > 1e-8 * g:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return 0.5 * pi * (
                (b1 + a1 * m0) / (2 * m0 * m0) + (b2 + a2 * m0) / (m0 * (m0 + p))
            )
    raise ValueError("ell_combo did not converge")


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390
    m1 = 1 - k * k
    epsilon = 1e-14
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / (m1 + epsilon))
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellk(k):
    a0 = 1.386294361120
    a1 = 0.096663442590
    a2 = 0.035900923830
    a3 = 0.037425637130
    a4 = 0.014511962120
    b0 = 0.50
    b1 = 0.124985935970
    b2 = 0.068802485760
    b3 = 0.033283553460
    b4 = 0.004417870120
    m1 = 1 - k * k
    epsilon = 1e-14
    return (a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) - (
        (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1 + epsilon)
    )


# Routine originally from:
# PyTransit: fast and easy exoplanet transit modelling in Python
# Copyright (C) 2010-2020  Hannu Parviainen
# Modified by Michael Hippke 2021, based on a GPL3 license
# Modifications:
# - Increased numerical stability for k=0.6 (exactly), see comments below
# - Fixed an error for missing ed[i], must be >= in "if (z >= 0.5 + abs(k - 0.5)"
# - Restricted to quadratic limb-darkening
# - Caching a few expensive values, e.g. from sqrt
# - Replaced some array generations with floats or copies
# - Overall speed improvements: ~10%
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult(zs, k, u1, u2):
    """Evaluates the transit model for an array of normalized distances.
    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u1, u2: float
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    # No k -> 0.5 snap for |k - 0.5| < 1e-4 any more (bug #30: up to 150 ppm);
    # the Case V/VI values below are overwritten by Case III/IV anyway
    INV_PI = 1 / pi
    k2 = k**2
    lzs = len(zs)
    epsilon = 1e-14
    flux = np.empty(lzs)
    le = np.empty(lzs)
    ld = np.empty(lzs)
    ed = np.empty(lzs)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    for i in range(lzs):
        z = zs[i]
        if abs(z - k) < 1e-6:
            z = k + 1e-6

        # The source is unocculted
        if z >= 1 + k or z < 0:
            flux[i] = 1
            le[i] = 0
            ld[i] = 0
            ed[i] = 0
            continue

        # The source is completely occulted
        elif k >= 1 and z <= k - 1:
            flux[i] = 0
            le[i] = 1
            ld[i] = 1
            ed[i] = 1
            continue

        z2 = z**2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k**2 - z2

        # Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(max(min((1 - k2 + z2) / (2 * z + epsilon), 1), -1))
            kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z + epsilon), 1), -1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # Edge of occulting body lies at the origin: special expressions in this case:
        if abs(z - k) <= 1e-4 * (z + k):
            # ! Table 3, Case V.:
            if k == 0.5:
                ld[i] = 1 / 3 - 4 * INV_PI / 9
                ed[i] = 3 / 32
            elif z > 0.5:
                ld[i] = (
                    1 / 3
                    + 16 * k / 9 * INV_PI * (2 * k2 - 1) * ellec(0.5 / k)
                    - (32 * k**4 - 20 * k2 + 3) / 9 * INV_PI / k * ellk(0.5 / k)
                )
                dist = sqrt((1 - x1) * (x2 - 1))
                ed[i] = (
                    1
                    / 2
                    * INV_PI
                    * (kap1 + k2 * (k2 + 2 * z2) * kap0 - (1 + 5 * k2 + z2) / 4 * dist)
                )
            elif z < 0.5 and 2 * k < 1:
                # ! Table 3, Case VI. (needs modulus 2k < 1):
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * ellec(2 * k) + (1 - 4 * k2) * ellk(2 * k)
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III. Written as the complement of Case IV below
        # (z < 1 - k): the former condition (z >= 0.5 + |k - 0.5|, or
        # k > 0.5 and |1 - k| <= z < k) is the same set mathematically, but
        # 0.5 + |k - 0.5| can round one ulp above 1 - k, and z = 1 - k then
        # matched neither case (ld, ed left uninitialised, bug #29)
        if z >= 1 - k and z < 1 + k:
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)

            # Numerical stability: if k=0.06 then q is close to 1, ellpicb goes to hell
            if (1 - q) < 1e-8:
                q -= 1e-8
            ld[i] = (
                1
                / 9
                * INV_PI
                / sqrt(k * z)
                * (
                    ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)) * ellk(q)
                    + 4 * k * z * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / x1 * ellpicb((1 / x1 - 1), q)
                )
            )
            if z < k:
                ld[i] = ld[i] + 2 / 3
            ed[i] = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0))
                )
            )

        # Occulting body transits the source:
        if k <= 1 and z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1))  # re-calc because different condition
            if q >= 1:
                q = 1 - 1e-8  # float rounding guard; ld is overwritten by the special case below
            ld[i] = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * ellk(q)
                    + (1 - x1) * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / (x1) * ellpicb((x2 / (x1) - 1), q)
                )
            )
            if z < k:
                ld[i] = ld[i] + 2 / 3
            if abs(k + z - 1) < 1e-8:
                ld[i] = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed[i] = k2 / 2 * (k2 + 2 * z2)
        flux[i] = 1 - (c1 * le[i] + c2 * ld[i] + u2 * ed[i]) * omega

        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if flux[i] > 1:
                flux[i] = 1
    return flux


# Fixed iteration count of the vectorised Bulirsch loop (P3b). 5 steps pass
# ell_combo's 1e-8 test for kc >= ~0.45 (q <= ~0.9); lanes with smaller kc
# (<= 0.5 % of the on-star samples for any k, mostly near second contact)
# fail the test after the last step and are recomputed with the scalar
# ell_combo_kc, so the result is always the converged one.
ELL_BATCH_ITERATIONS = 5

# Floor of the complementary modulus kc (P15). kc = 0 only at z = 1 - k
# exactly (or below it by rounding), where K and Pi diverge
# logarithmically but A K + B E + C Pi stays finite; with kc >= 1e-150 the
# result is off by O(kc^2 log kc), i.e. not at all, and the AGM converges
# in ~15 steps (scalar fallback).
KC_MIN = 1e-150

# Smallest |z - k| passed to the batch (occult_hybrid, pass 1), see there.
Z_K_MIN = 1e-150


# ---------------------------------------------------------------------------
# P14: interior series for bodies fully on the star (z + k < 1)
#
# With quadratic limb darkening I(mu) = (1 - u1 - u2) + (u1 + 2 u2) mu - u2 mu^2,
# mu = sqrt(1 - r^2), the blocked flux of a disk of radius k fully on the star
# needs only the disk means <mu^2> = 1 - z^2 - k^2 / 2 (exact) and <mu>:
#   F = 1 - k^2 Omega [(1 - u1 - u2) + (u1 + 2 u2) <mu> - u2 <mu^2>].
# For <mu>, write a point of the disk as z + rho (cos phi, sin phi), rho <= k:
#   mu = sqrt(w) sqrt(1 - eps),  w = 1 - z^2,  eps = (2 z rho cos phi + rho^2) / w,
#   sqrt(1 - eps) = 1 - sum_{m>=1} b_m eps^m,  b_m = |binom(1/2, m)| > 0,
#   <eps^m> = v^m sum_{l even <= m} G_{m,l} z^l k^(2m-l),  v = 1 / w,
#   G_{m,l} = binom(m, l) binom(l, l/2) 2 / (2m - l + 2) >= 0
# (disk average: <cos^l phi> = binom(l, l/2) / 2^l for even l, 0 for odd l;
# <rho^n> = 2 k^n / (n + 2)). On the star |eps| <= e = (2 z k + k^2) / w < 1,
# so the series converges uniformly and every averaged term is >= 0.
# Truncated after m = M (M odd), S_M = sqrt(w) P_M(v) with P_M a polynomial of
# degree M in v (z^l = (1 - 1/v)^(l/2) expanded), and
#   0 <= S_M - <mu> <= sqrt(w) sum_{m>M} b_m <eps^m>
#                   <= sqrt(w) b_{M+1} <eps^(M+1)> / (1 - e) =: bound(z)
# (|eps|^m <= |eps|^(M+1) e^(m-M-1), b_m non-increasing, M + 1 even so
# <|eps|^(M+1)> = <eps^(M+1)>). bound(z) increases with z (sqrt(w) v^(M+1),
# z^l and 1 / (1 - e) all do), so checking it at a single cut z_c covers all
# z < z_c: occult_hybrid uses the series for 0 <= z < z_c, where
# k^2 |Omega (u1 + 2 u2)| bound(z_c) <= SERIES_TOL, and the exact kernel
# elsewhere. Independent check (mpmath disk quadrature) in the unit tests.
# ---------------------------------------------------------------------------
SERIES_M = 11          # truncation order (odd); P_M has degree 11 in v
SERIES_TOL = 1e-8      # max. flux error of the series (0.01 ppm)
SERIES_E_MAX = 0.99    # cap of e = (2 z k + k^2) / (1 - z^2) at the cut


def _series_tables(M):
    """(alpha, b_{M+1}, G_{M+1, l} for l = 0, 2, ..., M+1) of the P14 series,
    exact rational arithmetic. alpha[j, p] is the coefficient of v^j k^(2p)
    in P_M(v) = 1 - sum_{m=1}^{M} b_m v^m sum_l G_{m,l} z^l k^(2m-l)."""
    from fractions import Fraction
    from math import comb, factorial

    def b(m):
        r = Fraction(1)
        for i in range(m):
            r *= Fraction(1, 2) - i
        return abs(r / factorial(m))

    def G(m, l):
        return Fraction(comb(m, l) * comb(l, l // 2) * 2, 2 * m - l + 2)

    A = [[Fraction(0)] * (M + 1) for _ in range(M + 1)]
    A[0][0] = Fraction(1)
    for m in range(1, M + 1):
        for l in range(0, m + 1, 2):
            g = b(m) * G(m, l)
            p = m - l // 2                      # k^(2m - l) = k^(2p)
            for i in range(l // 2 + 1):         # z^l = (1 - 1/v)^(l/2)
                A[m - i][p] -= g * comb(l // 2, i) * (-1) ** i
    alpha = np.array([[float(x) for x in row] for row in A])
    g_next = np.array([float(G(M + 1, l)) for l in range(0, M + 2, 2)])
    return alpha, float(b(M + 1)), g_next


SERIES_ALPHA, SERIES_B_NEXT, SERIES_G_NEXT = _series_tables(SERIES_M)
# <eps^(M+1)> / e^(M+1) at z >> k (lambda = 1, see _series_bound)
SERIES_GAMMA_1 = SERIES_G_NEXT[-1] * 0.5 ** (SERIES_M + 1)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _series_bound(z, k):
    """Upper bound of S_M - <mu> >= 0 at 0 <= z < 1 - k (see above):
    sqrt(w) b_{M+1} <eps^(M+1)> / (1 - e). Written as
    <eps^(M+1)> = e^(M+1) sum_l G_l (lam/2)^l (1 - lam)^(M+1-l),
    lam = 2 z / (2 z + k), so that no power over- or underflows."""
    w = 1 - z * z
    e = (2 * z * k + k * k) / w
    if not (e < 1):
        return 1e300  # (no inf: fastmath assumes finite values)
    lam = 2 * z / (2 * z + k)
    h2 = 0.25 * lam * lam
    g2 = (1 - lam) * (1 - lam)
    n = len(SERIES_G_NEXT)                      # (M + 1) / 2 + 1 terms
    # s = sum_i G_{2i} h2^i g2^(n-1-i): Horner in h2, g2 powers on the way
    s = 0.0
    gp = 1.0
    for i in range(n - 1, -1, -1):
        s = s * h2 + SERIES_G_NEXT[i] * gp
        gp *= g2
    em = 1.0
    for _ in range(SERIES_M + 1):
        em *= e
    return np.sqrt(w) * SERIES_B_NEXT * em * s / (1 - e)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _series_z_of_e(e, k):
    """z >= 0 with (2 z k + k^2) / (1 - z^2) = e (negative: no such z)"""
    return (np.sqrt(k * k * (1 - e) + e * e) - k) / e


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _series_zcut(k, tol_mu):
    """Cut z_c of the P14 series: _series_bound(z_c, k) <= tol_mu, verified
    at the returned value (so, by monotonicity, for every z < z_c), and
    usually >= 0.9 tol_mu (z_c within ~0.1 % of the largest such z).
    Returns 0.0 if no z qualifies (no series).

    Search in e = (2 z k + k^2) / (1 - z^2) (z(e) in closed form). Start
    from the model bound(e) ~ b gamma_1 sqrt(min(1, 2k/e)) e^(M+1) / (1 - e)
    (lam -> 1, w ~ 2k/e near the limb), then Newton steps in log-log space
    with the model slope D = M + 1 + e / (1 - e) toward 0.97 tol_mu; 1-3
    bound evaluations (~0.1 us)."""
    if not (k > 0 and k < 1 and tol_mu > 0):
        return 0.0
    tol_mu = min(tol_mu, 1e300)
    target = 0.97 * tol_mu
    e = min((target / (SERIES_B_NEXT * SERIES_GAMMA_1)) ** (1.0 / (SERIES_M + 1)), SERIES_E_MAX)
    D = (SERIES_M + 1) + e / (1 - e)
    e = min(e * ((1 - e) / np.sqrt(min(1.0, 2 * k / e))) ** (1 / D), SERIES_E_MAX)
    best = 0.0
    for it in range(4):
        z = _series_z_of_e(e, k)
        if not (z > 0):  # e below its value k^2 at z = 0: no series
            break
        bnd = _series_bound(z, k)
        if bnd <= tol_mu:
            if z > best:
                best = z
            if bnd >= 0.9 * tol_mu or e >= SERIES_E_MAX or not (bnd > 0):
                break
        D = (SERIES_M + 1) + e / (1 - e)
        e = min(e * (target / bnd) ** (1 / D), SERIES_E_MAX)
    return best


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _series_coeffs(k, scale):
    """scale * coefficients a_j(k) of P_M(v) (Horner in k^2 over alpha)"""
    k2 = k * k
    n = SERIES_ALPHA.shape[0]
    a = np.empty(n)
    for j in range(n):
        s = 0.0
        for p in range(n - 1, -1, -1):
            s = s * k2 + SERIES_ALPHA[j, p]
        a[j] = s * scale
    return a


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_series_batch(zb, m, a, A0, A1, flux):
    """P14 interior series for zb[:m] (all 0 <= z < z_c < 1 - k):
    flux = A0 + A1 z^2 - sqrt(w) P(v), P of degree SERIES_M = 11 in v = 1/w
    with the (scaled) coefficients a; clamped at 1. Unrolled Horner so that
    LLVM vectorises the loop (a loop over a[] would not be); ~0.9 ns per
    sample. error_model="numpy": w > 0 here."""
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    a4 = a[4]
    a5 = a[5]
    a6 = a[6]
    a7 = a[7]
    a8 = a[8]
    a9 = a[9]
    a10 = a[10]
    a11 = a[11]
    for j in range(m):
        z2 = zb[j] * zb[j]
        w = 1 - z2
        v = 1 / w
        P = a11
        P = P * v + a10
        P = P * v + a9
        P = P * v + a8
        P = P * v + a7
        P = P * v + a6
        P = P * v + a5
        P = P * v + a4
        P = P * v + a3
        P = P * v + a2
        P = P * v + a1
        P = P * v + a0
        f = A0 + A1 * z2 - np.sqrt(w) * P
        flux[j] = f if f < 1 else 1.0


assert SERIES_M == 11 and SERIES_ALPHA.shape == (12, 12)  # _occult_series_batch is unrolled for 11


# ---------------------------------------------------------------------------
# P21b: per-call polynomial interpolant of the limb band 1 - k <= z < 1 + k
#
# On the limb the exact kernel costs ~10.5 ns per lane (elliptic integrals,
# two arccos, ~12 divisions). For fixed (k, u1, u2) the deficit
# D(z) = (c1 lex + c2 ldx + u2 edx) omega is a smooth function of z on the
# band except at its ends: near first contact (h = 1 + k - z -> 0) it is a
# series in sqrt(h) starting at h^(3/2), near second contact
# (g = z - (1 - k) -> 0) D_t + a g + b g^(3/2) + ... plus g^2 ln g terms
# from the elliptic integrals. So the band is split at z = 1 and each half
# is interpolated in its own variable, u = sqrt(h / k) (outer half, z >= 1)
# or u = sqrt(g / k) (inner half), both in [0, 1], by a polynomial of
# degree LIMB_N in x = 2 u - 1. In u the outer half is analytic and the
# inner half has a u^4 ln u endpoint singularity (error ~N^-8). The
# conditions are the values at the Lobatto nodes x_j = cos(pi j / N),
# except that the nodes next to x = -1 are replaced by conditions known
# for free: outer D = dD/du = d2D/du2 = 0 at u = 0 (D ~ u^3), so nodes
# N-1, N-2 are dropped; inner D = D_t (second contact, closed form) and
# dD/du = 0 at u = 0, so node N-1 (kc ~ 0.012: a scalar fallback lane) is
# dropped. That leaves LIMB_NODES = 20 exact lanes per call (z = 1 is
# shared; one of them, inner node N-2, needs the scalar fallback); a fixed
# matrix per half turns the values into monomial coefficients (Horner per
# lane, ~2 ns, vectorised).
# Measured (scratch_p21b/acc3.py, mpmath reference for lex / edx, the
# converged kernel for ldx; dense z incl. 1e-14 from both contacts): max
# component error 2.2e-10 (k = 0.1), 1.2e-9 (k = 0.2), 2.5e-9 (k = 0.25),
# i.e. <= ~7e-9 in flux for any physical LD (test_limb_interpolant: <= 1e-8
# vs mpmath quadrature and occult_exact); near the contacts the
# interpolant is closer to mpmath than the double closed forms of lex (up
# to 7e-9 off within 1e-14 of first contact). Used by occult_hybrid (not
# occult_exact) for k <= LIMB_K_MAX when at least LIMB_MIN_LANES samples of
# a call lie on the band (the build costs ~0.35 us); tried only if
# len(zs) k >= LIMB_MIN_LANES LIMB_FILL (else the classification would
# cost for nothing, e.g. for Earth-sized moons).
# ---------------------------------------------------------------------------
LIMB_N = 12             # degree per half (the kernels below are unrolled for 12)
LIMB_K_MAX = 0.25       # validated range of k (error grows ~k^2)
LIMB_MIN_LANES = 64     # fewer band samples per call: exact kernel
LIMB_FILL = 0.5         # try the interpolant only if len(zs) * k >= LIMB_MIN_LANES * LIMB_FILL
LIMB_W = 16             # coefficient slots per half in _limb_coeffs' output (>= LIMB_N + 1)
LIMB_NO = LIMB_N - 2    # outer nodes x_0 .. x_{N-3}
LIMB_NI = LIMB_N - 1    # inner nodes x_0 .. x_{N-2}
LIMB_NODES = LIMB_NO + LIMB_NI - 1   # exact lanes (z = 1 shared): 20, a multiple of 4
                                     # (a scalar remainder lane of the batch costs ~25 ns)


def _limb_tables(N):
    """(u_j^2 at the Lobatto nodes x_j = cos(pi j / N), u = (x + 1) / 2;
    transposed matrices MO, MI: MO[j, i] = monomial coefficient i (in x) per
    unit value at outer node j (j < LIMB_NO; D, D', D'' = 0 at x = -1),
    MI[j, i] likewise for inner node j (j < LIMB_NI) and, at j = LIMB_NI,
    per unit second-contact value D_t (D' = 0 at x = -1))"""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    u = (x + 1) / 2

    def deriv_row(d):        # d-th derivative of x^i at x = -1
        r = np.zeros(N + 1)
        for i in range(d, N + 1):
            c = 1.0
            for q in range(d):
                c *= i - q
            r[i] = c * (-1.0) ** (i - d)
        return r

    V = np.vander(x, N + 1, increasing=True)
    no, ni = N - 2, N - 1
    A_o = np.vstack([V[:no], V[N], deriv_row(1), deriv_row(2)])
    A_i = np.vstack([V[:ni], V[N], deriv_row(1)])
    MO = np.linalg.inv(A_o)[:, :no]          # right-hand side zero in the other rows
    MI = np.linalg.inv(A_i)[:, :ni + 1]
    return u * u, np.ascontiguousarray(MO.T), np.ascontiguousarray(MI.T)


LIMB_U2, LIMB_MO, LIMB_MI = _limb_tables(LIMB_N)
assert LIMB_N == 12 and LIMB_NODES % 4 == 0  # _occult_limb_batch is unrolled for 12


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _limb_coeffs(k, u1, u2):
    """Monomial coefficients in x (outer half [0:N+1], inner half
    [LIMB_W:LIMB_W+N+1]) of the P21b limb interpolant of the deficit
    1 - flux for (k, u1, u2): exact kernel at the nodes (non-converged
    lanes scalar), closed form at second contact."""
    N = LIMB_N
    W = LIMB_W
    NO = LIMB_NO
    NI = LIMB_NI
    nn = LIMB_NODES
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    k2 = k * k
    # lanes: z = 1 (x_0 of both halves), outer x_1..x_{NO-1}, inner x_1..x_{NI-1}
    buf = np.empty(3 * nn)
    zn = buf[:nn]
    fn = buf[nn:2 * nn]
    dv = buf[2 * nn:]
    zn[0] = 1.0
    for j in range(1, NO):
        zn[j] = 1 + k - k * LIMB_U2[j]
    for j in range(1, NI):
        zn[NO - 1 + j] = 1 - k + k * LIMB_U2[j]
    _occult_exact_batch(zn, nn, k, c1, c2, u2, omega, fn, dv)
    for j in range(nn):
        if not dv[j] <= 0:
            fn[j] = _occult_point(zn[j], k, u1, u2)
    # second contact z = 1 - k in closed form (Case IV with z + k = 1)
    ldx_t = (2 / 3 / pi) * np.arccos(1 - 2 * k) - (4 / 9 / pi) * np.sqrt(k * (1 - k)) * (3 + 2 * k - 8 * k2)
    edx_t = k2 / 2 * (k2 + 2 * (1 - k) * (1 - k))
    d_t = (c1 * k2 + c2 * ldx_t + u2 * edx_t) * omega
    # coefficients = M @ conditions, as reductions per coefficient (44 ns;
    # row updates of both halves: 130 ns)
    coef = np.zeros(2 * W)
    for i in range(N + 1):
        d0 = 1 - fn[0]
        so = LIMB_MO[0, i] * d0
        si = LIMB_MI[0, i] * d0 + LIMB_MI[NI, i] * d_t
        for j in range(1, NO):
            so += LIMB_MO[j, i] * (1 - fn[j])
        for j in range(1, NI):
            si += LIMB_MI[j, i] * (1 - fn[NO - 1 + j])
        coef[i] = so
        coef[W + i] = si
    return coef


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_limb_batch(zl, m, k, coef, flux):
    """P21b limb interpolant for zl[:m] (all 1 - k <= z < 1 + k): flux =
    1 - D(z), clamped at 1; coef from _limb_coeffs(). Unrolled Horner with
    the half selected per lane, so that LLVM vectorises the loop."""
    inv_k = 1 / k
    o0 = coef[0]; o1 = coef[1]; o2 = coef[2]; o3 = coef[3]; o4 = coef[4]
    o5 = coef[5]; o6 = coef[6]; o7 = coef[7]; o8 = coef[8]; o9 = coef[9]
    o10 = coef[10]; o11 = coef[11]; o12 = coef[12]
    i0 = coef[16]; i1 = coef[17]; i2 = coef[18]; i3 = coef[19]; i4 = coef[20]
    i5 = coef[21]; i6 = coef[22]; i7 = coef[23]; i8 = coef[24]; i9 = coef[25]
    i10 = coef[26]; i11 = coef[27]; i12 = coef[28]
    for j in range(m):
        t = zl[j] - 1
        out = t >= 0
        r = ((k - t) if out else (k + t)) * inv_k
        x = 2 * np.sqrt(max(r, 0.0)) - 1
        P = o12 if out else i12
        P = P * x + (o11 if out else i11)
        P = P * x + (o10 if out else i10)
        P = P * x + (o9 if out else i9)
        P = P * x + (o8 if out else i8)
        P = P * x + (o7 if out else i7)
        P = P * x + (o6 if out else i6)
        P = P * x + (o5 if out else i5)
        P = P * x + (o4 if out else i4)
        P = P * x + (o3 if out else i3)
        P = P * x + (o2 if out else i2)
        P = P * x + (o1 if out else i1)
        P = P * x + (o0 if out else i0)
        f = 1 - P
        flux[j] = f if f < 1 else 1.0


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_exact_batch(zb, m, k, c1, c2, u2, omega, flux, dev):
    """Exact Mandel-Agol flux for the m on-star samples zb[:m] (P3b);
    branch-free, so LLVM vectorises it.

    Per sample: Case IV (z < 1 - k) or Case III (else; the limb), the
    coefficients A, B, C of A K + B E + C Pi selected per case, then the
    Bulirsch iteration of ell_combo with a fixed ELL_BATCH_ITERATIONS steps
    and no early exit (iterating past convergence is harmless: the AGM
    state only scales by 2 per step). flux[j] holds
    1 - (c1 lex + c2 ldx + u2 edx) omega (unclamped); dev[j] > 0 marks a
    lane whose last step fails ell_combo's 1e-8 test (recompute it with the
    scalar code).

    Since P16 the limb's area terms lex, edx (two arccos, Case III) are
    computed here for every lane with the branch-free trig.acos_poly and
    selected for limb lanes, instead of a scalar libm loop in the caller
    (16.4 ns per limb sample; acos_poly is <= 1 ulp from libm). Their
    divisors are floored at 1e-300 so that interior lanes (z = 0 or k = 0
    possible) stay finite under fastmath.

    Operands in closed form (P15; Agol, Luger & Foreman-Mackey 2020, eq. 41):
    with s = z + k, d = z - k, sg = sign(d)
      Case IV: kc^2 = (1 - s)(1 + s) / (1 - d^2), p = sqrt(n + 1) = s / |d|
      Case III: kc^2 = (s - 1)(s + 1) / (4 z k),  p = 1 / |d|
      C = -3 x3 / x1 = 3 s / d;  C / p = 3 sg (IV) or 3 s sg (III);
      prefactor 2 / (9 pi sqrt(den)) in both cases (den = 1 - d^2 or 4 z k).
    No modulus q, no 1 - q^2 cancellation, no q -> 1 - 1e-8 nudges (which
    were up to 2.6e-8 off in ldx at z -> 1 - k), 3 divisions and 2 square
    roots fewer per sample, and exact down to |z - k| = 1 ulp (the old
    z -> k + 1e-6 nudge is gone; only z == k exactly is moved by one ulp in
    occult_hybrid).

    error_model="numpy" is required for vectorisation: numba's default
    ("python") inserts a zero-divisor check before every division, and that
    branch stops LLVM. All divisors are > 0 here (z != k, den > 0 on the
    star, m0 >= 1, p > 0)."""
    INV_PI = 1 / pi
    k2 = k * k
    for j in range(m):
        z = zb[j]
        z2 = z * z
        s = z + k
        d = z - k
        x1 = d * d
        x2 = s * s
        x3 = k2 - z2
        inner = z < 1 - k
        # Case IV (inner) / Case III (limb): kc^2 = num / den
        num = (1 - s) * (1 + s) if inner else (s - 1) * (s + 1)
        den = 1 - x1 if inner else 4 * z * k
        inv_sden = 1 / sqrt(den)
        kc = max(sqrt(max(num, 0.0)) * inv_sden, KC_MIN)
        sg = 1.0 if d > 0 else -1.0
        inv_ad = 1 / abs(d)
        p = s * inv_ad if inner else inv_ad
        a2 = 3 * s * sg * inv_ad
        b2 = 3 * sg if inner else 3 * s * sg
        A = (1 - 5 * z2 + k2 + x3 * x3) if inner else ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2))
        B = den * (z2 + 7 * k2 - 4)
        # A K + B E + C Pi: ell_combo's iteration, one reciprocal
        # r = 1 / (m0 p) per step instead of two divisions
        a1 = A + B
        b1 = A + B * kc * kc
        e = kc
        m0 = 1.0
        dd = 0.0
        for it in range(ELL_BATCH_ITERATIONS):
            r = 1 / (m0 * p)
            f1 = a1
            a1 = a1 + b1 * (p * r)
            b1 = 2 * (b1 + f1 * kc)
            inv_p = m0 * r
            f2 = a2
            a2 = a2 + b2 * inv_p
            g = e * inv_p
            b2 = 2 * (b2 + f2 * g)
            p = g + p
            g = m0
            m0 = kc + m0
            dd = abs(g - kc) - 1e-8 * g
            kc = 2 * sqrt(e)
            e = kc * m0
        # (b1 + a1 m0) / (2 m0^2) + (b2 + a2 m0) / (m0 (m0 + p)), one division
        mp = m0 + p
        ell = 0.25 * pi * ((b1 + a1 * m0) * mp + 2 * m0 * (b2 + a2 * m0)) / (m0 * m0 * mp)
        ldx = (2 / 9 * INV_PI) * inv_sden * ell + (2 / 3 if z < k else 0.0)
        # Case III area terms (P16): computed for all lanes, used on the limb
        kap1 = acos_poly(max(min((1 - k2 + z2) / max(2 * z, 1e-300), 1.0), -1.0))
        kap0 = acos_poly(max(min((k2 + z2 - 1) / max(2 * k * z, 1e-300), 1.0), -1.0))
        lex_l = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0.0))) * INV_PI
        edx_l = 0.5 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0.0)))
        lex = k2 if inner else lex_l
        edx = k2 / 2 * (k2 + 2 * z2) if inner else edx_l
        flux[j] = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega
        dev[j] = dd


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_point(z, k, u1, u2):
    """Unclamped exact Mandel-Agol flux at one on-star z (z != k,
    0 <= z < 1 + k, not fully occulted), scalar ell_combo_kc. Same cases and
    closed-form operands as _occult_exact_batch (P15): Case IV for
    z < 1 - k, else Case III. Used for the batch's non-converged lanes."""
    INV_PI = 1 / pi
    k2 = k**2
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    z2 = z**2
    s = z + k
    d = z - k
    x1 = d * d
    x2 = s * s
    x3 = k**2 - z2
    sg = 1.0 if d > 0 else -1.0
    if z < 1 - k:
        # Occulting body transits the source: Table 3, Case IV
        lex = k2
        edx = k2 / 2 * (k2 + 2 * z2)
        num = (1 - s) * (1 + s)
        den = 1 - x1
        A = 1 - 5 * z2 + k2 + x3 * x3
        p = s / abs(d)
        Cp = 3 * sg
    else:
        # Partly occults the source and crosses the limb: Table 3, Case III
        kap1 = arccos(max(min((1 - k2 + z2) / (2 * z), 1), -1))
        kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z), 1), -1))
        lex = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        edx = 1 / 2 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0)))
        num = (s - 1) * (s + 1)
        den = 4 * z * k
        A = (1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)
        p = 1 / abs(d)
        Cp = 3 * s * sg
    sden = sqrt(den)
    kc = max(sqrt(max(num, 0.0)) / sden, KC_MIN)
    ldx = 2 / 9 * INV_PI / sden * ell_combo_kc(kc, p, A, den * (z2 + 7 * k2 - 4), 3 * s / d, Cp)
    if z < k:
        ldx = ldx + 2 / 3
    return 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid(zs, k, u1, u2):
    """Transit model for an array of normalized distances, for any k: the
    exact Mandel & Agol model (see _occult_hybrid), with the P14 interior
    series wherever it is within SERIES_TOL = 1e-8 (0.01 ppm) of it."""
    return _occult_hybrid(zs, k, u1, u2, SERIES_TOL)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_exact(zs, k, u1, u2):
    """occult_hybrid() without the P14 interior series: the exact model for
    every sample (~6.5 instead of ~0.9 ns per interior sample)."""
    return _occult_hybrid(zs, k, u1, u2, 0.0)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid(zs, k, u1, u2, series_tol):
    """Exact Mandel & Agol transit model (quadratic limb darkening) for an
    array of normalized distances, for any k, with the P14 interior series
    for z < z_c (series_tol = 0: exact everywhere). See
    _occult_hybrid_samples() for the model; P12: _occult_hybrid_stretches()
    for inputs whose classes come in long stretches (e.g. the window samples
    of a time-sorted light curve, _long_stretches()), else (and for z == k
    or full occultations) the per-sample version.

    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u1, u2: float
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    z_c = _hybrid_zcut(k, u1, u2, series_tol)
    # P21b limb interpolant: with the series only (occult_exact stays exact)
    limb = series_tol > 0 and k > 0 and k <= LIMB_K_MAX and len(zs) * k >= LIMB_MIN_LANES * LIMB_FILL
    if len(zs) >= STRETCH_MIN_N and _long_stretches(zs, k, z_c):
        ok, flux = _occult_hybrid_stretches(zs, k, u1, u2, z_c, limb)
        if ok:
            return flux
    return _occult_hybrid_samples(zs, k, u1, u2, z_c, limb)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _hybrid_zcut(k, u1, u2, series_tol):
    """P14 series cut z_c for (k, LD) at flux tolerance series_tol (0: no
    series)"""
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    den = k * k * abs((u1 + 2 * u2) * omega)  # |d flux / d <mu>|
    return _series_zcut(k, series_tol / den if den > 1e-300 else 1e300) if series_tol > 0 else 0.0


# P12: stretch scan. Samples of one class (off the star, interior series,
# exact) come in long contiguous stretches when z varies smoothly. A
# stretch is extended by blocks of STRETCH_BLOCK samples with a vectorised
# count of the class condition (0.08 ns per sample; blocks of <= 32 compile
# to a slow unrolled form), then sample by sample.
STRETCH_BLOCK = 64
STRETCH_MIN_AVG = 64    # stretch path only if stretches are this long on average
STRETCH_STRIDE = 8      # _long_stretches() looks at every 8th sample
STRETCH_MIN_N = 2048    # shorter inputs: per-sample path. The stretch pass
                        # gains ~1 ns per sample on PLATO windows (~5400
                        # samples); on ~1200-sample windows (Kepler, 5x
                        # supersampled) it was 1-2 % slower end to end


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _long_stretches(zs, k, z_c):
    """True if the class (off star / series / exact) of every
    STRETCH_STRIDE-th sample changes rarely enough that the stretches
    average >= STRETCH_MIN_AVG samples (cheap estimate; a wrong guess only
    costs speed)"""
    n_z = len(zs)
    one_k = 1 + k
    changes = 0
    prev = -1
    for i in range(0, n_z, STRETCH_STRIDE):
        z = zs[i]
        c = 0 if (z >= one_k or z < 0) else (1 if z < z_c else 2)
        changes += 1 if c != prev else 0
        prev = c
    return changes * STRETCH_MIN_AVG <= n_z


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_off(zs, j, n_z, one_k):
    """First index >= j of a sample with 0 <= z < one_k (end of an
    off-star stretch)"""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= 0) & (blk[q] < one_k) else 0
        if c:
            break
        j += STRETCH_BLOCK
    while j < n_z and (zs[j] >= one_k or zs[j] < 0):
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_series(zs, j, n_z, z_c, k):
    """First index >= j of a sample that is not an interior-series sample
    (0 <= z < z_c, z != k)"""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= 0) & (blk[q] < z_c) & (abs(blk[q] - k) >= Z_K_MIN) else 0
        if c < STRETCH_BLOCK:
            break
        j += STRETCH_BLOCK
    while j < n_z and zs[j] < z_c and zs[j] >= 0 and abs(zs[j] - k) >= Z_K_MIN:
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_exact(zs, j, n_z, z_c, one_k, k, full, km1):
    """First index >= j of a sample that is not an exact-model sample
    (z_c <= z < one_k, z != k, not fully occulted); with z_c = 1 - k,
    one_k = 1 + k: of the limb band (P21b)"""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= z_c) & (blk[q] < one_k) & (abs(blk[q] - k) >= Z_K_MIN) & ((not full) | (blk[q] > km1)) else 0
        if c < STRETCH_BLOCK:
            break
        j += STRETCH_BLOCK
    while j < n_z and zs[j] >= z_c and zs[j] < one_k and abs(zs[j] - k) >= Z_K_MIN and not (full and zs[j] <= km1):
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid_stretches(zs, k, u1, u2, z_c, limb):
    """_occult_hybrid_samples() by stretches (P12): (ok, flux). Off-star
    stretches are filled with 1 and interior-series stretches (z < z_c)
    evaluated in place (no gather, no index scatter); exact-model stretches
    are copied into one buffer for the vectorised batch, as before; with
    limb (P21b) limb-band stretches are evaluated in place by the
    interpolant if they hold >= LIMB_MIN_LANES samples together (else they
    join the batch). ok =
    False (the caller then uses the per-sample version) at a sample with
    z == k (one-ulp nudge) or a fully occulted one (k >= 1). Same values as
    the per-sample version up to rounding of the series (in place vs
    gathered: <= 2.2e-16)."""
    k2 = k**2
    n_z = len(zs)
    flux = np.empty(n_z)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    kk = k2 * omega
    one_k = 1 + k
    full = k >= 1
    km1 = k - 1
    idx = np.empty(n_z, dtype=np.int64)
    zb = np.empty(n_z)
    # P21b: limb stretches 1 - k <= z < 1 + k, kept as ranges [rl[2r], rl[2r+1])
    one_mk = 1 - k if limb else one_k
    rl = np.empty(n_z + 2 if limb else 0, dtype=np.int64)
    n_r = 0
    n_l = 0
    m = 0
    have_a = False
    a = np.empty(0)
    A0 = 0.0
    A1 = -kk * u2
    i = 0
    while i < n_z:
        z = zs[i]
        j = i + 1
        if z >= one_k or z < 0:                    # off the star: flux 1
            j = _end_off(zs, j, n_z, one_k)
            fr = flux[i:j]
            for q in range(j - i):
                fr[q] = 1.0
        elif abs(z - k) < Z_K_MIN or (full and z <= km1):
            return False, flux
        elif z < z_c:                              # interior series, in place
            j = _end_series(zs, j, n_z, z_c, k)
            if not have_a:
                a = _series_coeffs(k, kk * c2)
                A0 = 1 - kk * ((1 - u1 - u2) - u2 * (1 - k2 / 2))
                have_a = True
            _occult_series_batch(zs[i:j], j - i, a, A0, A1, flux[i:j])
        elif z >= one_mk:                          # limb band (P21b): range
            j = _end_exact(zs, j, n_z, one_mk, one_k, k, full, km1)
            rl[2 * n_r] = i
            rl[2 * n_r + 1] = j
            n_r += 1
            n_l += j - i
        else:                                      # exact (z_c <= z < one_mk): gather
            j = _end_exact(zs, j, n_z, z_c, one_mk, k, full, km1)
            cnt = j - i
            zr = zs[i:j]
            zbr = zb[m:m + cnt]
            ir = idx[m:m + cnt]
            for q in range(cnt):
                zbr[q] = zr[q]
                ir[q] = i + q
            m += cnt
        i = j
    use_limb = n_l >= LIMB_MIN_LANES
    if use_limb:                                   # interpolant, in place
        coef = _limb_coeffs(k, u1, u2)
        for r in range(n_r):
            lo = rl[2 * r]
            hi = rl[2 * r + 1]
            _occult_limb_batch(zs[lo:hi], hi - lo, k, coef, flux[lo:hi])
    else:                                          # few limb samples: exact batch
        for r in range(n_r):
            lo = rl[2 * r]
            cnt = rl[2 * r + 1] - lo
            zr = zs[lo:lo + cnt]
            zbr = zb[m:m + cnt]
            ir = idx[m:m + cnt]
            for q in range(cnt):
                zbr[q] = zr[q]
                ir[q] = lo + q
            m += cnt
    if m > 0:
        fb = np.empty(m)
        dev = np.empty(m)
        _occult_exact_batch(zb, m, k, c1, c2, u2, omega, fb, dev)
        # non-converged lanes (scalar), clamp, scatter
        for j in range(m):
            f = fb[j]
            if not dev[j] <= 0:
                f = _occult_point(zb[j], k, u1, u2)
            if f > 1:
                f = 1
            flux[idx[j]] = f
    return True, flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid_samples(zs, k, u1, u2, z_c, limb):
    """_occult_hybrid() with a per-sample pass 1 (before P12 this was
    _occult_hybrid; still used for inputs without long stretches, e.g.
    unsorted time); z_c from _hybrid_zcut(); limb: use the P21b limb
    interpolant if >= LIMB_MIN_LANES samples lie on 1 - k <= z < 1 + k
    (only for 0 < k <= LIMB_K_MAX). The model:

    Until P3 this switched to a "linear interpolation" correction of the
    small-body approximation for k <= 0.05 (z <= 0.65 ... 0.98); that zone
    existed only for speed and was off by up to ~1.1 ppm (bug #28). It is
    gone: the batched exact model below is cheaper than it was.

    Batched (P3b): pass 1 classifies the samples (off star, fully occulted)
    and collects the remaining on-star z; pass 2 computes their exact flux
    in the vectorised _occult_exact_batch(); pass 3 recomputes non-converged
    lanes with the scalar _occult_point() and clamps (the limb's arccos
    terms are part of the batch since P16, with the vectorisable
    trig.acos_poly). Case III is simply "not Case IV", which closes a 1-ulp
    gap at z = 1 - k where the old conditions matched neither case and
    reused the previous sample's terms (bug #29).
    Since P15 the elliptic operands come in closed form from (z, k), exact
    at the contact points: the z -> k + 1e-6 and q -> 1 - 1e-8 nudges and
    the closed-form special case for |k + z - 1| < 1e-8 (up to ~2.6e-8 off
    in the limb-darkening term) are gone.
    Since P14, interior samples with z < z_c (body fully on the star, away
    from the limb) use the series of _occult_series_batch() (~0.9 ns instead
    of ~6.5 ns per sample); z_c is chosen per call such that the series is
    within series_tol (occult_hybrid: SERIES_TOL = 1e-8, 0.01 ppm) of the
    exact flux, by a rigorous tail bound (see the comment block above
    SERIES_M); series_tol = 0 (occult_exact) disables it. All other on-star
    samples take the exact path above.
    """
    # k is used as given: the old snap k -> 0.5 for |k - 0.5| < 1e-4 was up
    # to 150 ppm off (bug #30); the closed-form operands (P15) are exact at
    # k = 0.5 and z -> k (non-converged lanes take the scalar fallback)
    k2 = k**2
    n_z = len(zs)
    flux = np.empty(n_z)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    # Pass 1: classify; collect the on-star samples: interior series
    # (z < z_c, P14) from the front of zb / idx, exact model from the back;
    # with limb (P21b) the band 1 - k <= z < 1 + k goes to zb[n_z:] (right
    # after the exact ones, so that it can join their batch if it is short)
    kk = k2 * omega
    one_mk = 1 - k if limb else 1 + k      # 1 + k: no sample is a limb sample
    nb = 2 * n_z if limb else n_z
    idx = np.empty(nb, dtype=np.int64)
    zb = np.empty(nb)
    ms = 0
    me = n_z
    ml = n_z
    for i in range(n_z):
        z = zs[i]
        # The closed-form operands (P15) are exact down to |z - k| = 1 ulp;
        # only z == k itself (C = 3 (z + k) / (z - k)) moves, by one ulp
        # (k + k 2^-52 is at least the next float above k). Z_K_MIN keeps
        # 1 / |z - k| finite for k < ~1e-134, where ulps are smaller.
        if abs(z - k) < Z_K_MIN:
            z = k + max(k * 2.220446049250313e-16, Z_K_MIN)

        # Source is unocculted
        if z >= 1 + k or z < 0:
            flux[i] = 1
            continue

        # The source is completely occulted
        elif k >= 1 and z <= k - 1:
            flux[i] = 0
            continue

        if z < z_c:
            zb[ms] = z
            idx[ms] = i
            ms += 1
        elif z >= one_mk:
            zb[ml] = z
            idx[ml] = i
            ml += 1
        else:
            me -= 1
            zb[me] = z
            idx[me] = i
    n_l = ml - n_z
    use_limb = n_l >= LIMB_MIN_LANES
    m = (n_z if use_limb else ml) - me

    # Pass 2a: interior series (vectorised), written straight to flux
    fb = np.empty(max(ms, m, n_l if use_limb else 0))
    if ms > 0:
        a = _series_coeffs(k, kk * c2)
        A0 = 1 - kk * ((1 - u1 - u2) - u2 * (1 - k2 / 2))
        _occult_series_batch(zb, ms, a, A0, -kk * u2, fb)
        for j in range(ms):
            flux[idx[j]] = fb[j]

    # Pass 2b: exact flux of the other on-star samples (vectorised)
    zx = zb[me:me + m]
    ix = idx[me:me + m]
    dev = np.empty(m)
    _occult_exact_batch(zx, m, k, c1, c2, u2, omega, fb, dev)

    # Pass 3: non-converged lanes (scalar), clamp, scatter. The limb's
    # arccos terms are in the batch since P16.
    for j in range(m):
        f = fb[j]
        if not dev[j] <= 0:
            f = _occult_point(zx[j], k, u1, u2)
        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if f > 1:
            f = 1
        flux[ix[j]] = f

    # Pass 4 (P21b): limb band by the per-call interpolant (clamped)
    if use_limb:
        coef = _limb_coeffs(k, u1, u2)
        zl = zb[n_z:ml]
        il = idx[n_z:ml]
        _occult_limb_batch(zl, n_l, k, coef, fb)
        for j in range(n_l):
            flux[il[j]] = fb[j]
    return flux
