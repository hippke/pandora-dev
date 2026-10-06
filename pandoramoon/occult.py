import numpy as np
from numpy import sqrt, pi, arccos, abs, log, ceil
from numba import jit


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
    kc = sqrt(1 - q * q)
    a1 = A + B
    b1 = A + B * kc * kc
    p = sqrt(n + 1)
    a2 = C
    b2 = C / p
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
    if abs(k - 0.5) < 1e-4:
        k = 0.5

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
            elif z < 0.5:
                # ! Table 3, Case VI.:
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * ellec(2 * k) + (1 - 4 * k2) * ellk(2 * k)
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III:
        if (z >= 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z >= abs(1 - k) and z < k
        ):
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
# ell_combo's 1e-8 test for q <= ~0.9; lanes with q closer to 1 (<= 0.5 % of
# the on-star samples for any k, mostly near second contact) fail the test
# after the last step and are recomputed with the scalar ell_combo, so the
# result is always the converged one.
ELL_BATCH_ITERATIONS = 5


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_exact_batch(zb, m, k, c1, c2, u2, omega, flux, dev):
    """Exact Mandel-Agol flux without the limb's arccos terms, for the m
    on-star samples zb[:m] (P3b); branch-free, so LLVM vectorises it.

    Per sample: Case IV (z < 1 - k) or Case III (else; the limb), the
    coefficients A, B, C of A K + B E + C Pi selected per case, then the
    Bulirsch iteration of ell_combo with a fixed ELL_BATCH_ITERATIONS steps
    and no early exit (iterating past convergence is harmless: the AGM
    state only scales by 2 per step). flux[j] holds
    1 - (c1 lex + c2 ldx + u2 edx) omega with lex = edx = 0 for limb samples
    (the caller adds their arccos terms); dev[j] > 0 marks a lane whose last
    step fails ell_combo's 1e-8 test (recompute it with the scalar code).

    error_model="numpy" is required for vectorisation: numba's default
    ("python") inserts a zero-divisor check before every division, and that
    branch stops LLVM. All divisors are > 0 here (z > 0, x1 >= 1e-12 by the
    z -> k + 1e-6 nudge, 1 - x1 > 0 for z < 1 - k, m0 >= 1, p > 0)."""
    INV_PI = 1 / pi
    k2 = k * k
    for j in range(m):
        z = zb[j]
        z2 = z * z
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k2 - z2
        inner = z < 1 - k
        # Case IV (inner) / Case III (limb): select operands, divide once
        num = x2 - x1 if inner else 1 - x1
        den = 1 - x1 if inner else 4 * z * k
        q = sqrt(num / den)
        # Case IV: float rounding guard; Case III: q -> 1 at the lower z
        # boundary makes the iteration diverge, push it slightly below 1
        q = min(q, 1 - 1e-8) if inner else (q - 1e-8 if 1 - q < 1e-8 else q)
        inv_x1 = 1 / x1
        n = (x2 if inner else 1.0) * inv_x1 - 1
        A = (1 - 5 * z2 + k2 + x3 * x3) if inner else ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2))
        B = (1 - x1) * (z2 + 7 * k2 - 4) if inner else 4 * k * z * (z2 + 7 * k2 - 4)
        C = -3 * x3 * inv_x1
        pref = (2 / 9 * INV_PI if inner else 1 / 9 * INV_PI) / sqrt(den if inner else k * z)
        # A K(q) + B E(q) + C Pi(n, q): ell_combo's iteration, one
        # reciprocal r = 1 / (m0 p) per step instead of two divisions
        kc = sqrt(1 - q * q)
        a1 = A + B
        b1 = A + B * kc * kc
        p = sqrt(n + 1)
        a2 = C
        b2 = C / p
        e = kc
        m0 = 1.0
        d = 0.0
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
            d = abs(g - kc) - 1e-8 * g
            kc = 2 * sqrt(e)
            e = kc * m0
        ell = 0.5 * pi * ((b1 + a1 * m0) / (2 * m0 * m0) + (b2 + a2 * m0) / (m0 * (m0 + p)))
        ldx = pref * ell + (2 / 3 if z < k else 0.0)
        lex = k2 if inner else 0.0
        edx = k2 / 2 * (k2 + 2 * z2) if inner else 0.0
        flux[j] = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega
        dev[j] = d


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_point(z, k, u1, u2):
    """Unclamped exact Mandel-Agol flux at one on-star z (after the z -> k +
    1e-6 nudge; 0 < z < 1 + k, not fully occulted), scalar ell_combo. Same
    cases as _occult_exact_batch: Case IV for z < 1 - k, else Case III.
    Used for the batch's non-converged lanes and the k + z = 1 special case."""
    INV_PI = 1 / pi
    k2 = k**2
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    z2 = z**2
    x1 = (k - z) ** 2
    x2 = (k + z) ** 2
    x3 = k**2 - z2
    if z < 1 - k:
        # Occulting body transits the source: Table 3, Case IV
        lex = k2
        edx = k2 / 2 * (k2 + 2 * z2)
        if abs(k + z - 1) < 1e-8:
            ldx = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                k * (1 - k)) * (3 + 2 * k - 8 * k2)
        else:
            q = sqrt((x2 - x1) / (1 - x1))
            if q >= 1:
                q = 1 - 1e-8  # float rounding guard
            ldx = 2 / 9 * INV_PI / sqrt(1 - x1) * ell_combo(
                q, x2 / x1 - 1, 1 - 5 * z2 + k2 + x3 * x3, (1 - x1) * (z2 + 7 * k2 - 4),
                -3 * x3 / x1)
            if z < k:
                ldx = ldx + 2 / 3
    else:
        # Partly occults the source and crosses the limb: Table 3, Case III
        kap1 = arccos(max(min((1 - k2 + z2) / (2 * z), 1), -1))
        kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z), 1), -1))
        lex = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        edx = 1 / 2 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0)))
        q = sqrt((1 - x1) / 4 / z / k)
        if (1 - q) < 1e-8:
            q -= 1e-8
        ldx = 1 / 9 * INV_PI / sqrt(k * z) * ell_combo(
            q, 1 / x1 - 1, (1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2),
            4 * k * z * (z2 + 7 * k2 - 4), -3 * x3 / x1)
        if z < k:
            ldx = ldx + 2 / 3
    return 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid(zs, k, u1, u2):
    """Evaluates the exact transit model (Mandel & Agol 2002, quadratic limb
    darkening) for an array of normalized distances, for any k.

    Until P3 this switched to a "linear interpolation" correction of the
    small-body approximation for k <= 0.05 (z <= 0.65 ... 0.98); that zone
    existed only for speed and was off by up to ~1.1 ppm (bug #28). It is
    gone: the batched exact model below is cheaper than it was.

    Batched (P3b): pass 1 classifies the samples (off star, fully occulted,
    the z + k = 1 special case) and collects the remaining on-star z; pass 2
    computes their exact flux in the vectorised _occult_exact_batch(); pass 3
    recomputes non-converged lanes with the scalar _occult_point(), adds the
    limb's arccos terms and clamps.
    Formulas, guards and nudges as before (P3a); Case III is now simply
    "not Case IV", which closes a 1-ulp gap at z = 1 - k where the old
    conditions matched neither case and reused the previous sample's terms.

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
    if abs(k - 0.5) < 1e-4:
        k = 0.5

    INV_PI = 1 / pi
    k2 = k**2
    n_z = len(zs)
    flux = np.empty(n_z)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    # Pass 1: classify; collect the on-star samples that need the exact model
    idx = np.empty(n_z, dtype=np.int64)
    zb = np.empty(n_z)
    m = 0
    for i in range(n_z):
        z = zs[i]
        if abs(z - k) < 1e-6:
            z = k + 1e-6

        # Source is unocculted
        if z >= 1 + k or z < 0:
            flux[i] = 1
            continue

        # The source is completely occulted
        elif k >= 1 and z <= k - 1:
            flux[i] = 0
            continue

        # Case IV special case k + z = 1 (closed form for ldx)
        if z < 1 - k and abs(k + z - 1) < 1e-8:
            f = _occult_point(z, k, u1, u2)
            if f > 1:
                f = 1
            flux[i] = f
            continue

        zb[m] = z
        idx[m] = i
        m += 1

    # Pass 2: exact flux of all collected samples (vectorised), limb terms excluded
    fb = np.empty(m)
    dev = np.empty(m)
    _occult_exact_batch(zb, m, k, c1, c2, u2, omega, fb, dev)

    # Pass 3: non-converged lanes, limb arccos terms, clamp
    for j in range(m):
        z = zb[j]
        if not dev[j] <= 0:
            f = _occult_point(z, k, u1, u2)
        elif z >= 1 - k:
            # Star partially occulted and the occulting object crosses the limb
            z2 = z * z
            x1 = (k - z) ** 2
            x2 = (k + z) ** 2
            kap1 = arccos(max(min((1 - k2 + z2) / (2 * z), 1), -1))
            kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z), 1), -1))
            lex = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
            edx = 1 / 2 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                    - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0)))
            f = fb[j] - (c1 * lex + u2 * edx) * omega
        else:
            f = fb[j]
        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if f > 1:
            f = 1
        flux[idx[j]] = f
    return flux
