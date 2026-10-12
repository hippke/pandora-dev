"""Bayesian fits of Pandora models: priors, evidences, posteriors (public API of the nss sampler).

    import pandoramoon as pandora
    from pandoramoon import NestedSampler, Uniform, LogUniform, Normal

    like = pandora.Likelihood(time, flux, flux_err, epoch_distance=365.26)
    priors = dict(
        q1=Uniform(0, 1), q2=Uniform(0, 1),                # or u1=..., u2=... (e.g. Normal: measured LD)
        R_star=Uniform(0.4 * R_sun, 1.2 * R_sun),          # Pandora's units: m, kg, days, degrees
        per_bary=Uniform(360.25, 370.25), a_bary=Uniform(210, 220), r_planet=Uniform(0.07, 0.11),
        b_bary=Uniform(0, 1), t0_bary=11.0, t0_bary_offset=Uniform(-0.05, 0.05),
        M_planet=Uniform(1.5e27, 2.5e27),
        r_moon=Uniform(0.01, 0.07), per_moon=Uniform(0, 50), tau_moon=Uniform(0, 1),
        Omega_moon=Uniform(0, 180), i_moon=Uniform(0, 180), M_moon=Uniform(1, 1e27))
        # a number = fixed; w_bary, ecc_bary, ecc_moon, w_moon default to 0 (circular orbits)
    sampler = NestedSampler(like, priors, log_dir="fit_out")
    results = sampler.run()                  # planet+moon AND planet-only fit (for the Bayes factor)
    sampler.print_results()
    results["logz"], results["dlogz"], results["samples"], results["planet"]["logz"]
    sampler.plot_corner()

Design (SAMPLING.md Round 15):
- The 20 parameters of Pandora's model (`Likelihood` call order) can be free (a prior object) or fixed
  (a number). Priors: Uniform, LogUniform, Normal (optionally truncated); limb darkening either as Kipping's
  (q1, q2) or as (u1, u2) with any of these priors (truncated to the physical region, renormalised), and a
  correlation between Normal u1 and u2 (`ld_correlation`).
- Moon fits sample the moon's period and phase in the phase-at-transit coordinates (REPARAMETRIZATION.md)
  by default, with an exact weight: the prior, evidence and posterior are those of the user's parameters.
  This needs a per_moon prior with an upper bound <= the planet period and tau_moon ~ Uniform(0, 1);
  otherwise (or with moon_phase=False) the native parameters are sampled. Omega_moon ~ Uniform(lo, lo + 180)
  is sampled on a periodic double cover (Omega + 180 deg with the phase + 1/2 is the same orbit) and
  reported folded.
- The sampler is pandoramoon.nss.fit (independent compiled runs on all cores, merged, self-checking). The
  compiled likelihood takes the data and prior constants as an argument: it is compiled once per process
  and reused for every fit and data set.
- Results follow UltraNest's result dictionary (logz, logzerr, samples, weighted_samples, posterior,
  maxlikelihood, ncall, paramnames, ...), plus the self-check (passed, history) and, for moon fits, the
  planet-only fit ("planet") and the Bayes factor dlogz = logZ(moon) - logZ(planet).
- Files (log_dir) like UltraNest: info/results.json, info/post_summary.csv, chains/equal_weighted_post.txt,
  chains/weighted_post.npz, plots/corner.pdf; the planet-only fit in log_dir/planet/. Every finished run is
  stored in runs/, so an interrupted fit continues with resume="resume".
"""
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
from numba import njit

from pandoramoon import nss
from pandoramoon.pandora import pandora_loglike, pandora_loglike_planet
from pandoramoon.reparam import min_physical_period

R_SUN = 696_342_000.0
LOGZERO = -1e300
PARAMS = ['u1', 'u2', 'R_star', 'per_bary', 'a_bary', 'r_planet', 'b_bary', 'w_bary', 'ecc_bary', 't0_bary',
          't0_bary_offset', 'M_planet', 'r_moon', 'per_moon', 'tau_moon', 'Omega_moon', 'i_moon', 'ecc_moon',
          'w_moon', 'M_moon']
IDX = {n: i for i, n in enumerate(PARAMS)}
PLANET_NAMES = PARAMS[:12]
MOON_NAMES = PARAMS[12:]
DEFAULTS = dict(w_bary=0.0, ecc_bary=0.0, ecc_moon=0.0, w_moon=0.0)
# parameters that do not change the planet-only model (they act only through the moon's orbit)
PLANET_IRRELEVANT = ('R_star', 'M_planet')


# ------------------------------------------------------------------------------------------------ priors

class Prior:
    kind = -1

    def ppf(self, u):
        raise NotImplementedError

    def support(self):
        return self.lo, self.hi

    def as_dict(self):
        return dict(type=type(self).__name__, **{k: float(v) for k, v in self.__dict__.items()})

    def __repr__(self):
        return f"{type(self).__name__}(" + ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()) + ")"


class Uniform(Prior):
    """flat prior on [lo, hi]"""
    kind = 0

    def __init__(self, lo, hi):
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"Uniform({lo}, {hi}): need finite lo < hi")
        self.lo, self.hi = lo, hi

    def par(self):
        return (self.lo, self.hi, 0.0, 0.0)


class LogUniform(Prior):
    """prior flat in log(x) on [lo, hi], lo > 0"""
    kind = 1

    def __init__(self, lo, hi):
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(hi) and 0 < lo < hi):
            raise ValueError(f"LogUniform({lo}, {hi}): need 0 < lo < hi < inf")
        self.lo, self.hi = lo, hi

    def par(self):
        return (math.log(self.lo), math.log(self.hi), 0.0, 0.0)


class Normal(Prior):
    """Gaussian prior N(mean, sigma), optionally truncated to [lo, hi] (renormalised)"""
    kind = 2

    def __init__(self, mean, sigma, lo=-np.inf, hi=np.inf):
        mean, sigma, lo, hi = float(mean), float(sigma), float(lo), float(hi)
        if not (np.isfinite(mean) and sigma > 0 and np.isfinite(sigma) and hi > lo):
            raise ValueError(f"Normal({mean}, {sigma}, {lo}, {hi}): need sigma > 0 and lo < hi")
        self.mean, self.sigma, self.lo, self.hi = mean, sigma, lo, hi
        clo, chi = _norm_cdf((lo - mean) / sigma), _norm_cdf((hi - mean) / sigma)
        if not chi - clo > 1e-12:
            raise ValueError(f"Normal({mean}, {sigma}, {lo}, {hi}): (almost) no prior mass in [lo, hi]")

    def par(self):
        return (self.mean, self.sigma, _norm_cdf((self.lo - self.mean) / self.sigma),
                _norm_cdf((self.hi - self.mean) / self.sigma))

    def support(self, nsig=12.0):
        """finite range holding all but ~1e-32 of the mass (for range computations only)"""
        return max(self.lo, self.mean - nsig * self.sigma), min(self.hi, self.mean + nsig * self.sigma)


def _norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# Pandora's valid range of each parameter (pandora.params_valid); priors reaching beyond it are truncated to it
# and renormalised (Round 17), else their mass outside would be silently lost (likelihood -inf there)
PHYSICAL = dict(q1=(0.0, 1.0), q2=(0.0, 1.0), R_star=(0.0, math.inf), per_bary=(0.0, math.inf),
                a_bary=(0.0, math.inf), M_planet=(0.0, math.inf), per_moon=(0.0, math.inf), r_planet=(0.0, math.inf),
                r_moon=(0.0, math.inf), M_moon=(0.0, math.inf), ecc_bary=(0.0, 1.0), ecc_moon=(0.0, 1.0))


def _mass_in(p, lo, hi):
    """prior mass of p in [lo, hi]"""
    if isinstance(p, Uniform):
        return max(0.0, min(hi, p.hi) - max(lo, p.lo)) / (p.hi - p.lo)
    if isinstance(p, LogUniform):
        a, b = max(lo, p.lo), min(hi, p.hi)
        return max(0.0, math.log(b / a)) / math.log(p.hi / p.lo) if b > a > 0 else 0.0
    _, _, c, d = p.par()
    a = _norm_cdf((max(lo, p.lo) - p.mean) / p.sigma)
    b = _norm_cdf((min(hi, p.hi) - p.mean) / p.sigma)
    return max(0.0, b - a) / (d - c)


def _truncate(p, lo, hi):
    """p restricted to [lo, hi] and renormalised (same type)"""
    a, b = max(lo, p.lo), min(hi, p.hi)
    if isinstance(p, Uniform):
        return Uniform(a, b)
    if isinstance(p, LogUniform):
        return LogUniform(a, b)
    return Normal(p.mean, p.sigma, a, b)


# ------------------------------------------------------------------------------- compiled building blocks

@njit(cache=True)
def _ndtri(p):
    """inverse standard normal CDF (Acklam's approximation + one Halley step; |error| ~ 3e-10)"""
    if not (p > 0.0 and p < 1.0):
        return -math.inf if p <= 0.0 else math.inf
    a1, a2, a3 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02
    a4, a5, a6 = 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b1, b2, b3 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02
    b4, b5 = 6.680131188771972e+01, -1.328068155288572e+01
    c1, c2, c3 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00
    c4, c5, c6 = -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


@njit(cache=True)
def _ppf(kind, a, b, c, d, u):
    """unit-cube coordinate u -> parameter value (inverse CDF of the prior)"""
    if kind == 0:
        return a + u * (b - a)
    if kind == 1:
        return math.exp(a + u * (b - a))
    # kind 2: truncated normal (a = mean, b = sigma, c, d = CDF at the bounds)
    return a + b * _ndtri(c + u * (d - c))


@njit(cache=True)
def _logpdf(kind, a, b, c, d, x):
    """log prior density at x (kind / parameters as in _ppf); -inf outside the support"""
    if kind == 0:
        if x < a or x > b:
            return -math.inf
        return -math.log(b - a)
    if kind == 1:
        if x <= 0.0 or math.log(x) < a or math.log(x) > b:
            return -math.inf
        return -math.log(x) - math.log(b - a)
    z = (x - a) / b
    lo = a + b * _ndtri(c) if c > 0.0 else -math.inf
    hi = a + b * _ndtri(d) if d < 1.0 else math.inf
    if x < lo or x > hi:
        return -math.inf
    return -0.5 * z * z - math.log(b * math.sqrt(2.0 * math.pi)) - math.log(d - c)


@njit(cache=True)
def _tab_ppf(TX, TC, u):
    """repartitioned coordinate (Round 21, NEW_LEADS L1): piecewise-linear inverse of the tabulated CDF (TC at the
    nodes TX); returns (x, log q(x)) with q the density this map induces (piecewise constant, exact)"""
    lo = 0
    hi = TC.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if TC[mid] <= u:
            lo = mid
        else:
            hi = mid
    dc = TC[hi] - TC[lo]
    dx = TX[hi] - TX[lo]
    return TX[lo] + (u - TC[lo]) * dx / dc, math.log(dc / dx)


@njit(cache=True)
def _to_native(per_bary, t0_bary, t0_off, phi_c, delta, u_m, e_c, CDF, M_MIN, Q, P_MAX):
    """phase-at-transit coordinates -> (per_moon, tau, log w); as reparam.Reparam.to_native, with the
    constants as arguments (one compiled function for every prior)"""
    lo, hi = 0, CDF.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if CDF[mid] > u_m:
            hi = mid
        else:
            lo = mid + 1
    m = M_MIN + lo
    N = m + delta
    per_moon = per_bary / N
    t_c = t0_bary + t0_off + e_c * per_bary
    x = t_c * N / per_bary - phi_c
    tau = x - math.floor(x)
    if per_moon > P_MAX:
        return per_moon, tau, LOGZERO
    return per_moon, tau, math.log(per_bary / P_MAX * Q * m * (m + 1.0) / (N * N))


# D = the data / prior tuple passed through nss: ( 0-10: Likelihood._tail without the cache,
#   11 model (0 planet, 1 moon), 12 base[20] (fixed values), 13 nat_p, 14 nat_c, 15 nat_k, 16 nat_par (k, 4),
#   17 ld_mode (0: q1/q2 -> u1/u2, 1: u1/u2 with the triangle cut, 2: correlated normal u1/u2, 3: fixed u),
#   18 ld_c[2] (cube indices, mode 2), 19 ld_par[6] (m1, m2, L11, L21, L22, log norm),
#   20 mp (moon-phase coordinates on/off), 21 mp_c[3] (cube indices of phi_c, delta, u_m), 22 CDF,
#   23 mp_f[6] (M_MIN, Q, P_MAX, e_c, p_lo, log P_MAX), 24 pm_k (per_moon prior kind), 25 pm_par[4] )

@njit(cache=True)
def _theta(u, D):
    """unit cube -> the 20 Pandora parameters and the log weight (LOGZERO: outside the prior)"""
    th = D[12].copy()
    lw = 0.0
    nat_p, nat_c, nat_k, nat_par = D[13], D[14], D[15], D[16]
    tab, TX, TC = D[26], D[27], D[28]
    for j in range(nat_p.shape[0]):
        if tab[j] < 0:
            th[nat_p[j]] = _ppf(nat_k[j], nat_par[j, 0], nat_par[j, 1], nat_par[j, 2], nat_par[j, 3], u[nat_c[j]])
        else:
            # posterior repartitioning: sample from the tabulated mixture q, weight pi(x) / q(x) (Z, posterior exact)
            x, lq = _tab_ppf(TX[tab[j]], TC[tab[j]], u[nat_c[j]])
            th[nat_p[j]] = x
            lw += _logpdf(nat_k[j], nat_par[j, 0], nat_par[j, 1], nat_par[j, 2], nat_par[j, 3], x) - lq
    ld_mode = D[17]
    if ld_mode == 0:
        q1 = th[0]
        q2 = th[1]
        if not (q1 >= 0.0 and q1 <= 1.0 and q2 >= 0.0 and q2 <= 1.0):
            return th, LOGZERO
        sq = math.sqrt(q1)
        th[0] = 2.0 * sq * q2
        th[1] = sq * (1.0 - 2.0 * q2)
    elif ld_mode == 1 or ld_mode == 2:
        lp = D[19]
        if ld_mode == 2:
            z1 = _ndtri(u[D[18][0]])
            z2 = _ndtri(u[D[18][1]])
            th[0] = lp[0] + lp[2] * z1
            th[1] = lp[1] + lp[3] * z1 + lp[4] * z2
        if not (th[0] >= 0.0 and th[0] + th[1] <= 1.0 and th[0] + 2.0 * th[1] >= 0.0):
            return th, LOGZERO
        lw += lp[5]
    if D[20] == 1:
        mc = D[21]
        mf = D[23]
        pm, tau, logw = _to_native(th[3], th[9], th[10], u[mc[0]], u[mc[1]], u[mc[2]], mf[3], D[22], int(mf[0]),
                                   mf[1], mf[2])
        if not logw > LOGZERO or pm < mf[4]:
            return th, LOGZERO
        pp = D[25]
        lpdf = _logpdf(D[24], pp[0], pp[1], pp[2], pp[3], pm)
        if not lpdf > -math.inf:
            return th, LOGZERO
        lw += logw + lpdf + mf[5]
        th[13] = pm
        th[14] = tau
    return th, lw


@njit(cache=True)
def _loglike(u, D):
    """log L'(u) = log L(theta(u)) + log w(u): the function nss samples on the unit cube"""
    th, lw = _theta(u, D)
    if not lw > LOGZERO:
        return LOGZERO
    if D[11] == 0:
        return lw + pandora_loglike_planet(th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], th[8], th[9],
                                           th[10], th[11], D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8],
                                           None, D[9], D[10])
    return lw + pandora_loglike(th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], th[8], th[9], th[10], th[11],
                                th[12], th[13], th[14], th[15], th[16], th[17], th[18], th[19], D[0], D[1], D[2], D[3],
                                D[4], D[5], D[6], D[7], D[8], None, D[9], D[10])


@njit(cache=True)
def _thetas(U, D):
    n = U.shape[0]
    out = np.empty((n, 20))
    lw = np.empty(n)
    for i in range(n):
        th, w = _theta(U[i], D)
        out[i] = th
        lw[i] = w
    return out, lw


# ------------------------------------------------------------------------------------------- the model

class _Model:
    """the compiled likelihood's constants for one model ("moon" or "planet") of a Likelihood and priors"""

    def __init__(self, like, priors, model, moon_phase, ld_correlation, repartition=None, repartition_weight=0.8):
        if like.cache is not None:
            raise ValueError("NestedSampler: Likelihood(cache=...) is not supported; use cache=None")
        self.model = model
        names = PLANET_NAMES + (MOON_NAMES if model == "moon" else [])
        ld = "q" if ("q1" in priors or "q2" in priors) else "u"
        if ld == "q" and ("u1" in priors or "u2" in priors):
            raise ValueError("give the limb darkening either as q1, q2 or as u1, u2, not both")
        spec = {}
        for n in names:
            key = {"u1": "q1", "u2": "q2"}.get(n, n) if ld == "q" else n
            if key in priors:
                spec[key] = priors[key]
            elif n in DEFAULTS:
                spec[key] = DEFAULTS[n]
            elif model == "planet" and n in PLANET_IRRELEVANT:
                spec[key] = None
            else:
                raise ValueError(f"no prior or value for {key!r}")
        if model == "planet":                     # R_star and M_planet do not change the planet-only model
            spec["R_star"] = R_SUN
            spec["M_planet"] = 1e27
        for k, v in spec.items():
            if not isinstance(v, Prior):
                try:
                    spec[k] = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{k}: a prior (Uniform, LogUniform, Normal) or a number, got {v!r}")
        self.truncated = {}
        for k, v in list(spec.items()):
            if isinstance(v, Prior) and k in PHYSICAL:
                lo, hi = PHYSICAL[k]
                if v.lo < lo or v.hi > hi:
                    frac = _mass_in(v, lo, hi)
                    if not frac > 1e-6:
                        raise ValueError(f"{k}: the prior {v!r} has (almost) no mass in Pandora's valid range [{lo}, {hi}]")
                    spec[k] = _truncate(v, lo, hi)
                    if frac < 1 - 1e-9:
                        self.truncated[k] = 1.0 - frac
                        import warnings
                        warnings.warn(f"{k}: {100 * (1 - frac):.3g} % of the prior {v!r} lies outside Pandora's valid range "
                                      f"[{lo}, {hi}]; the prior is truncated to it and renormalised", stacklevel=3)
        self.spec = spec
        self.ld = ld
        self.free = [k for k in spec if isinstance(spec[k], Prior)]
        base = np.zeros(20)
        if model == "planet":                     # Pandora's planet-only moon (orbit inside the planet)
            base[IDX["per_moon"]] = 1e-6
        for k, v in spec.items():
            if not isinstance(v, Prior):
                base[IDX[{"q1": "u1", "q2": "u2"}.get(k, k)]] = v
        # moon-phase coordinates?
        mp = False
        if model == "moon" and moon_phase:
            pm, tau = spec["per_moon"], spec["tau_moon"]
            mp = (isinstance(pm, Prior) and isinstance(tau, Uniform) and tau.lo == 0.0 and tau.hi == 1.0)
            if moon_phase is True and not mp:
                raise ValueError("moon_phase=True needs a per_moon prior and tau_moon = Uniform(0, 1) "
                                 "(use moon_phase='auto' or False to sample the native parameters)")
        self.moon_phase = mp
        cube, wrapped = [], []                    # names of the cube coordinates, periodic flags
        nat_p, nat_c, nat_k, nat_par = [], [], [], []
        self.double_cover = None
        ld_mode, ld_c, ld_par = (0 if ld == "q" else 1), [0, 0], [0.0] * 6
        corr = float(ld_correlation)
        if ld == "u":
            p1, p2 = spec["u1"], spec["u2"]
            if corr != 0.0:
                if not (isinstance(p1, Normal) and isinstance(p2, Normal) and np.isinf(p1.lo) and np.isinf(p1.hi)
                        and np.isinf(p2.lo) and np.isinf(p2.hi)):
                    raise ValueError("ld_correlation needs untruncated Normal priors on u1 and u2")
                if not -1.0 < corr < 1.0:
                    raise ValueError("ld_correlation must be in (-1, 1)")
                ld_mode = 2
                ld_par[:5] = [p1.mean, p2.mean, p1.sigma, corr * p2.sigma, p2.sigma * math.sqrt(1 - corr * corr)]
            elif not isinstance(p1, Prior) and not isinstance(p2, Prior):
                ld_mode = 3
        for k in spec:
            v = spec[k]
            if not isinstance(v, Prior):
                continue
            if mp and k in ("per_moon", "tau_moon"):
                continue
            if ld_mode == 2 and k in ("u1", "u2"):
                ld_c[0 if k == "u1" else 1] = len(cube)
                cube.append(k)
                wrapped.append(False)
                continue
            p = IDX[{"q1": "u1", "q2": "u2"}.get(k, k)]
            par = list(v.par())
            per = False
            if k == "Omega_moon" and isinstance(v, Uniform) and abs((v.hi - v.lo) - 180.0) < 1e-9:
                par = [v.lo, v.lo + 360.0, 0.0, 0.0]          # periodic double cover, reported folded
                per = True
                self.double_cover = v.lo
            elif k in ("tau_moon",) and isinstance(v, Uniform) and v.lo == 0.0 and v.hi == 1.0:
                per = True
            elif k in ("w_bary", "w_moon") and isinstance(v, Uniform) and abs((v.hi - v.lo) - 360.0) < 1e-9:
                per = True
            nat_p.append(p)
            nat_c.append(len(cube))
            nat_k.append(v.kind)
            nat_par.append(par)
            cube.append(k)
            wrapped.append(per)
        mp_c, CDF, mp_f, pm_k, pm_par = [0, 0, 0], np.zeros(1), [0.0] * 6, 0, [0.0] * 4
        self.p_lo = None
        self.F_excluded = 0.0
        if mp:
            pm = spec["per_moon"]
            plo_s, p_max = pm.support() if not isinstance(pm, Normal) else (max(pm.lo, 0.0), pm.hi)
            if not (np.isfinite(p_max) and p_max > 0 and plo_s >= 0):
                raise ValueError("per_moon prior: need a support within (0, P_max] with finite P_max")
            pb = spec["per_bary"]
            pb_min, pb_max = (pb.support() if isinstance(pb, Prior) else (pb, pb))
            pb_min = max(pb_min, 1e-9)
            if p_max > pb_min:
                raise ValueError(f"per_moon prior reaches {p_max} d > the shortest planet period {pb_min} d; the "
                                 "moon-phase coordinates need per_moon <= per_bary (use moon_phase=False)")

            def hi_of(k):
                v = spec[k]
                return v.support()[1] if isinstance(v, Prior) else v

            def lo_of(k):
                v = spec[k]
                return v.support()[0] if isinstance(v, Prior) else v
            p_phys = min_physical_period(hi_of("R_star"), max(lo_of("r_planet"), 0) + max(lo_of("r_moon"), 0),
                                         hi_of("M_planet") + hi_of("M_moon"))
            if plo_s > 0:
                p_lo = plo_s
            else:
                # below p_lo every moon of the prior box is unphysical (a_m < r_p + r_m): its likelihood is the
                # planet-only one, so that prior mass F is added back exactly with the planet fit's evidence
                p_lo = 0.6 * p_phys
                if not p_lo > 0:
                    raise ValueError("per_moon prior starting at 0 needs r_planet + r_moon > 0 in the prior box")
                self.F_excluded = float(np.exp(_logcdf_np(pm, p_lo)))
            self.p_lo = p_lo
            M_MIN = max(1, int(math.floor(pb_min / p_max)))
            M_MAX = int(math.ceil(pb_max / p_lo))
            if M_MAX - M_MIN > 50_000_000:
                raise ValueError("per_moon / per_bary prior range too wide for the moon-phase coordinates")
            ms = np.arange(M_MIN, M_MAX + 1, dtype=np.float64)
            qm = 1.0 / ms - 1.0 / (ms + 1.0)
            Qs = float(qm.sum())
            CDF = np.cumsum(qm) / Qs
            CDF[-1] = 1.0
            e_c = _ref_epoch(like, spec)
            self.e_c = e_c
            mp_c = [len(cube), len(cube) + 1, len(cube) + 2]
            cube += ["phi_c", "delta", "u_m"]
            wrapped += [True, True, False]
            mp_f = [float(M_MIN), Qs, float(p_max), float(e_c), float(p_lo), math.log(p_max)]
            pm_k, pm_par = pm.kind, list(pm.par())
        # LD normalisation (u1, u2 priors truncated to the physical triangle)
        if ld_mode in (1, 2):
            ld_par[5] = -math.log(self._ld_p_inside(spec, ld_mode, ld_par))
        self.cube_names = cube
        self.wrapped = wrapped
        self.ndim = len(cube)
        tail = like._tail
        self.D = (float(tail[0]), int(tail[1]), float(tail[2]), float(tail[3]), int(tail[4]),
                  np.ascontiguousarray(tail[5], dtype=np.float64), np.ascontiguousarray(tail[6], dtype=np.float64),
                  np.ascontiguousarray(tail[7], dtype=np.float64), float(tail[8]), bool(tail[10]), bool(tail[11]),
                  int(model == "moon"), base,
                  np.array(nat_p, dtype=np.int64), np.array(nat_c, dtype=np.int64), np.array(nat_k, dtype=np.int64),
                  np.array(nat_par, dtype=np.float64).reshape(-1, 4),
                  int(ld_mode), np.array(ld_c, dtype=np.int64), np.array(ld_par, dtype=np.float64),
                  int(mp), np.array(mp_c, dtype=np.int64), np.ascontiguousarray(CDF, dtype=np.float64),
                  np.array(mp_f, dtype=np.float64), int(pm_k), np.array(pm_par, dtype=np.float64),
                  *self._tables(cube, nat_c, repartition, repartition_weight))

    def _tables(self, cube, nat_c, repartition, beta, K=4096):
        """posterior repartitioning (Round 21, NEW_LEADS L1): for each parameter in repartition = {name: (mu, sigma)},
        the inverse CDF of q = (1 - beta) pi + beta N(mu, sigma) (truncated to pi's support) tabulated at K + 1
        quantile nodes. The map is the piecewise-linear interpolant of the table and its density is used in the
        weight, so Z and the posterior are exact for any table; the table only sets the efficiency."""
        self.repartition = {}
        tab = np.full(len(nat_c), -1, dtype=np.int64)
        TXs, TCs = [], []
        for j, c in enumerate(nat_c):
            name = cube[c]
            if not repartition or name not in repartition:
                continue
            p = self.spec[name]
            mu, sg = map(float, repartition[name])
            TX, TC = _mixture_table(p, mu, sg, float(beta), K)
            if TX is None:
                import warnings
                warnings.warn(f"repartition {name}: proposal N({mu}, {sg}) has no mass in the prior support; not used")
                continue
            tab[j] = len(TXs)
            TXs.append(TX)
            TCs.append(TC)
            self.repartition[name] = (mu, sg, float(beta))
        if TXs:
            n = max(len(x) for x in TXs)
            TX = np.array([np.pad(x, (0, n - len(x)), mode="edge") for x in TXs])
            TC = np.array([np.pad(x, (0, n - len(x)), mode="edge") for x in TCs])
        else:
            TX = np.zeros((1, 2))
            TC = np.array([[0.0, 1.0]])
        return tab, np.ascontiguousarray(TX), np.ascontiguousarray(TC)

    @staticmethod
    def _ld_p_inside(spec, ld_mode, ld_par, n=4_000_000):
        """prior mass of the (u1, u2) prior inside Pandora's physical triangle (Monte Carlo, fixed seed)"""
        rng = np.random.default_rng(12345)
        if ld_mode == 2:
            z = rng.standard_normal((n, 2))
            u1 = ld_par[0] + ld_par[2] * z[:, 0]
            u2 = ld_par[1] + ld_par[3] * z[:, 0] + ld_par[4] * z[:, 1]
        else:
            def draw(p):
                if not isinstance(p, Prior):
                    return np.full(n, p)
                return _ppf_np(p, rng.uniform(size=n))
            u1, u2 = draw(spec["u1"]), draw(spec["u2"])
        p_in = float(np.mean((u1 >= 0) & (u1 + u2 <= 1) & (u1 + 2 * u2 >= 0)))
        if p_in < 1e-3:
            raise ValueError(f"limb darkening prior: only {p_in:.2g} of its mass is physical")
        return p_in

    def native(self, U):
        """unit-cube points -> (theta (n, 20), log w (n,))"""
        return _thetas(np.ascontiguousarray(U, dtype=np.float64), self.D)

    def paramnames(self):
        return list(self.free)

    def to_params(self, U, fold=True):
        """unit-cube points -> values of the free parameters (columns as paramnames()), Omega folded"""
        th, _ = self.native(U)
        if fold and self.double_cover is not None:
            om = th[:, IDX["Omega_moon"]]
            hi = (om - self.double_cover) >= 180.0
            th[hi, IDX["Omega_moon"]] -= 180.0
            th[hi, IDX["tau_moon"]] = np.mod(th[hi, IDX["tau_moon"]] - 0.5, 1.0)
        cols = []
        for k in self.free:
            if k == "q1":
                cols.append((th[:, 0] + th[:, 1]) ** 2)
            elif k == "q2":
                s = th[:, 0] + th[:, 1]
                cols.append(np.where(s > 0, th[:, 0] / (2 * np.where(s > 0, s, 1.0)), 0.5))
            else:
                cols.append(th[:, IDX[k]])
        return np.column_stack(cols) if cols else np.empty((len(th), 0))


def _ppf_np(p, u):
    a, b, c, d = p.par()
    if p.kind == 0:
        return a + u * (b - a)
    if p.kind == 1:
        return np.exp(a + u * (b - a))
    return a + b * _ndtri_vec(c + u * (d - c))


@njit(cache=True)
def _ndtri_vec(p):
    out = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        out[i] = _ndtri(p[i])
    return out


def _logcdf_np(p, x):
    if isinstance(p, Uniform):
        return math.log(min(max((x - p.lo) / (p.hi - p.lo), 1e-300), 1.0))
    if isinstance(p, LogUniform):
        return math.log(min(max(math.log(x / p.lo) / math.log(p.hi / p.lo), 1e-300), 1.0)) if x > p.lo else -np.inf
    a, b, c, d = p.par()
    return math.log(max((_norm_cdf((x - a) / b) - c) / (d - c), 1e-300))


def _cdf_np(p, x):
    """prior CDF at an array x (Uniform / LogUniform / truncated Normal)"""
    x = np.asarray(x, dtype=float)
    if isinstance(p, Uniform):
        return np.clip((x - p.lo) / (p.hi - p.lo), 0.0, 1.0)
    if isinstance(p, LogUniform):
        return np.clip(np.log(np.maximum(x, 1e-300) / p.lo) / math.log(p.hi / p.lo), 0.0, 1.0)
    a, b, c, d = p.par()
    from math import erfc, sqrt
    z = (x - a) / b
    ph = 0.5 * np.vectorize(erfc)(-z / sqrt(2.0))
    return np.clip((ph - c) / (d - c), 0.0, 1.0)


def _mixture_table(p, mu, sg, beta, K):
    """nodes (x, CDF) of q = (1 - beta) p + beta N(mu, sg) truncated to p's support; (None, None) if the normal has
    no mass there. Note: an untruncated Normal prior is cut at +-12 sigma (mass ~1e-32 dropped)."""
    lo, hi = p.support()
    from math import erfc, sqrt
    Phi = lambda z: 0.5 * np.vectorize(erfc)(-np.asarray(z) / sqrt(2.0))
    glo, ghi = float(Phi((lo - mu) / sg)), float(Phi((hi - mu) / sg))
    if not ghi - glo > 1e-12 or not sg > 0:
        return None, None
    if isinstance(p, LogUniform):
        g1 = np.geomspace(lo, hi, 200_001)
    else:
        g1 = np.linspace(lo, hi, 200_001)
    g2 = mu + sg * np.linspace(-12, 12, 40_001)
    xg = np.unique(np.concatenate([g1, g2[(g2 > lo) & (g2 < hi)]]))
    F = (1 - beta) * _cdf_np(p, xg) + beta * (Phi((xg - mu) / sg) - glo) / (ghi - glo)
    F = (F - F[0]) / (F[-1] - F[0])
    t = np.linspace(0.0, 1.0, K + 1)
    X = np.interp(t, F, xg)
    X[0], X[-1] = lo, hi
    C = (1 - beta) * _cdf_np(p, X) + beta * (Phi((X - mu) / sg) - glo) / (ghi - glo)
    C = (C - C[0]) / (C[-1] - C[0])
    keep = np.r_[True, (np.diff(X) > 0) & (np.diff(C) > 0)]
    X, C = X[keep], C[keep]
    C[0], C[-1] = 0.0, 1.0
    return X, C


def _ref_epoch(like, spec):
    """reference transit epoch e_c of the moon-phase coordinates: the mean of the epochs with data"""
    def mid(k):
        v = spec[k]
        if isinstance(v, Normal):
            return v.mean
        return 0.5 * sum(v.support()) if isinstance(v, Prior) else v
    t0 = mid("t0_bary") + mid("t0_bary_offset")
    pb = mid("per_bary")
    ep = np.unique(np.floor((like.time - t0) / pb + 0.5).astype(np.int64))
    return int(round(float(np.mean(ep))))


# ------------------------------------------------------------------------------------- results, files

def _summary(samples, names):
    q = np.percentile(samples, [15.86553, 50.0, 84.13447], axis=0) if len(samples) else np.zeros((3, len(names)))
    return dict(mean=samples.mean(axis=0).tolist(), stdev=samples.std(axis=0).tolist(), median=q[1].tolist(),
                errlo=q[0].tolist(), errup=q[2].tolist())


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


_RUN_SCALARS = ("seed", "nlive", "logz", "logzerr", "H", "ncall", "niter", "wall", "wall_init", "ess", "maxL",
                "final_scale", "expand_rate", "nclusters", "nclusters_max", "nrecluster", "nfallback", "dirs",
                "insertion", "logx0", "init_acc", "init_draws", "n_plateau", "nstuck", "nmaxexp")


def _save_run(path, r):
    meta = {k: _jsonable(r[k]) for k in _RUN_SCALARS if k in r}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, dead_u=r["dead_u"], dead_L=r["dead_L"], logw=r["logw"], ins_u=r["ins_u"], mix=r["mix"],
                        meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def _load_run(path):
    z = np.load(path)
    r = json.loads(str(z["meta"]))
    for k in ("dead_u", "dead_L", "logw", "ins_u", "mix"):
        r[k] = z[k]
    return r


class _Store:
    """log_dir layout like UltraNest: info/, chains/, plots/ and runs/ (the stored nss runs, for resume)"""

    def __init__(self, path):
        self.path = path
        for d in ("info", "chains", "plots", "runs"):
            os.makedirs(os.path.join(path, d), exist_ok=True)

    def state_file(self):
        return os.path.join(self.path, "info", "fit_state.json")

    def read_state(self):
        try:
            return json.load(open(self.state_file()))
        except (OSError, ValueError):
            return None

    def write_state(self, st):
        tmp = self.state_file() + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_file())

    def run_path(self, nsteps, idx):
        return os.path.join(self.path, "runs", f"run{idx:05d}_nsteps{nsteps}.npz")

    def stored_runs(self):
        out = []
        for f in sorted(os.listdir(os.path.join(self.path, "runs"))):
            if f.startswith("run") and f.endswith(".npz") and ".tmp" not in f:
                idx = int(f[3:8])
                ns = int(f.split("nsteps")[1].split(".")[0])
                out.append((ns, idx, os.path.join(self.path, "runs", f)))
        return out


def _thin(w, max_n, seed=0):
    """indices and new weights of an unbiased thinning of weighted points to about max_n (Horvitz-Thompson):
    point i is kept with probability p_i = min(1, w_i / c) and then carries the weight w_i / p_i; c is chosen so
    that sum p_i = max_n. Points with large weight are always kept; estimates of means are unbiased."""
    if max_n is None or len(w) <= max_n:
        return np.arange(len(w)), w
    ws = np.sort(w)[::-1]
    cs = np.cumsum(ws)
    # find c: sum min(1, w/c) = max_n  <=>  k + (S - cs[k-1]) / c = max_n for the k largest kept with p = 1
    c = None
    for k in range(0, int(max_n)):
        rest = cs[-1] - (cs[k - 1] if k > 0 else 0.0)
        cc = rest / (max_n - k)
        if k == 0 or ws[k - 1] >= cc:
            if ws[k] <= cc:
                c = cc
                break
    if c is None:
        c = cs[-1] / max_n
    p = np.minimum(1.0, w / c)
    keep = np.random.default_rng(seed).uniform(size=len(w)) < p
    idx = np.nonzero(keep)[0]
    nw = w[idx] / p[idx]
    return idx, nw / nw.sum()


def _fit_model(m, nss_kw, store, resume_mode, settings_hash, verbose, label, max_weighted=None):
    """one nss.fit of a _Model, with the run store / resume; returns the UltraNest-like result dict"""
    resume = None
    callback = None
    if store is not None:
        st = store.read_state()
        if resume_mode == "resume" and st is not None:
            if st.get("hash") != settings_hash:
                raise ValueError(f"{store.path}: stored fit has different data, priors or settings; use "
                                 "resume='overwrite' or 'subfolder'")
            runs = []
            for ns, idx, path in store.stored_runs():
                try:
                    runs.append((ns, idx, _load_run(path)))
                except Exception:                 # e.g. a file cut by a crash: ignore it
                    pass
            resume = dict(runs=runs, N=st.get("N"), nesc=st.get("nesc", 0), skip=st.get("skip", []),
                          nlive=st.get("nlive"), nlesc=st.get("nlesc", 0))
            if verbose:
                print(f"  {label}: resuming with {len(runs)} stored runs (nsteps {st.get('N')})", flush=True)
        else:
            for _ns, _idx, path in store.stored_runs():
                os.remove(path)
            store.write_state(dict(hash=settings_hash, N=None, nesc=0))

        def callback(event, **kw):
            if event == "run":
                _save_run(store.run_path(kw["nsteps"], kw["idx"]), kw["r"])
            elif event == "state":
                st = store.read_state() or {}
                st.update(hash=settings_hash, N=kw["N"], nesc=kw["nesc"], nlive=kw.get("nlive"),
                          nlesc=kw.get("nlesc", 0),
                          skip=sorted(set(st.get("skip", [])) | set(kw.get("dropped_idx", []))))
                store.write_state(st)
                gone = set(kw.get("dropped_idx", []))
                for ns, idx, path in store.stored_runs():
                    if ns == kw["dropped"] or idx in gone:
                        os.remove(path)
        if resume is None or resume.get("N") is None:
            st = store.read_state() or {}
            st.update(hash=settings_hash, N=nss_kw.get("nsteps") or 4 * m.ndim, nesc=0)
            store.write_state(st)
            if resume is not None:
                resume["N"] = st["N"]

    def post_map(U):
        return m.to_params(U)
    t0 = time.perf_counter()
    r = nss.fit(_loglike, m.ndim, wrapped=m.wrapped, data=m.D, post_map=post_map, verbose=verbose, resume=resume,
                callback=callback, **nss_kw)
    wall = time.perf_counter() - t0
    names = m.paramnames()
    samples = m.to_params(r["samples_u"])
    # weighted samples: all points of the merged result runs (weights of the merge as one run)
    runs = r["runs"]
    _, lwl = nss.merge_weights(runs)
    w = np.exp(lwl - np.logaddexp.reduce(lwl))
    keep = w > 0
    U = np.concatenate([q["dead_u"] for q in runs])[keep]
    Lp = np.concatenate([q["dead_L"] for q in runs])[keep]
    w = w[keep] / w[keep].sum()
    n_weighted_all = len(w)
    ti, w = _thin(w, max_weighted, seed=1)
    U, Lp = U[ti], Lp[ti]
    _, lw = m.native(U)
    logl = Lp - lw                                # the user's log-likelihood (without the coordinate weight)
    pts = m.to_params(U)
    logz = float(r["logz"])
    H = float(np.sum(w * (logl - logz)))         # information gain (natural units), native prior
    ib = int(np.argmax(logl))
    hist = r["history"][-1]
    res = dict(
        paramnames=names, logz=logz, logzerr=float(r["logzerr"]), logzerr_bs=float(r["logzerr_boot"]),
        logz_single=float(np.mean([q["logz"] for q in runs])), ess=float(r["ess"]), H=H,
        Herr=float(math.sqrt(max(H, 0.0) / (r["nlive"] * len(runs)))),
        ncall=int(r["ncall"]), niter=int(sum(q["niter"] for q in runs)),
        posterior=_summary(samples, names), samples=samples,
        weighted_samples=dict(upoints=U, points=pts, weights=w, logl=logl),
        maxlikelihood=dict(logl=float(logl[ib]), point=pts[ib].tolist(), point_untransformed=U[ib].tolist()),
        insertion_order_test=dict(D=hist.get("ins_pooled_D"), p=hist.get("ins_pooled_p"),
                                  p_fisher=hist.get("ins_fisher_p")),
        passed=bool(r["passed"]), nsteps=int(r["nsteps"]), nlive=int(r["nlive"]), runs=len(runs),
        runs_check=len(r["runs_check"]), n_escalations=int(r["n_escalations"]), n_failed=int(r["n_failed"]),
        n_nlive_escalations=int(r.get("n_nlive_escalations", 0)), multimodal=bool(r.get("multimodal", False)),
        dlogz_check=r.get("dlnz_check"), ks_arms=r.get("ks_arms"), history=r["history"], wall=wall,
        t_compile=float(r["t_compile"]), model=m.model, moon_phase=bool(m.moon_phase),
        sampled_coordinates=list(m.cube_names), priors_truncated=dict(m.truncated),
        n_weighted_all=int(n_weighted_all),
        run_logz=[float(q["logz"]) for q in runs], run_logzerr=[float(q["logzerr"]) for q in runs],
        run_maxL=[float(q["maxL"]) for q in runs])
    if store is not None:
        st = store.read_state() or {}
        st.update(hash=settings_hash, N=int(r["nsteps"]), nesc=int(r["n_escalations"]), finished=True)
        store.write_state(st)
    return res


def _write_results(path, res, priors_repr):
    os.makedirs(os.path.join(path, "info"), exist_ok=True)
    os.makedirs(os.path.join(path, "chains"), exist_ok=True)
    small = {k: v for k, v in res.items() if k not in ("samples", "weighted_samples", "planet")}
    small["priors"] = priors_repr
    json.dump(_jsonable(small), open(os.path.join(path, "info", "results.json"), "w"), indent=1)
    names = res["paramnames"]
    p = res["posterior"]
    hdr = ",".join(f"{n}_{s}" for n in names for s in ("mean", "stdev", "median", "errlo", "errup"))
    row = ",".join(f"{p[s][j]!r}" for j in range(len(names)) for s in ("mean", "stdev", "median", "errlo", "errup"))
    open(os.path.join(path, "info", "post_summary.csv"), "w").write(hdr + "\n" + row + "\n")
    np.savetxt(os.path.join(path, "chains", "equal_weighted_post.txt"), res["samples"], header=" ".join(names),
               comments="")
    ws = res["weighted_samples"]
    np.savez_compressed(os.path.join(path, "chains", "weighted_post.npz"), names=np.array(names), points=ws["points"],
                        upoints=ws["upoints"], weights=ws["weights"], logl=ws["logl"])


def _data_hash(like):
    h = hashlib.sha1()
    for a in (like.time, like.inv_var, like.lin_terms):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(like._tail[:5] + (like._tail[8],) + like._tail[10:]).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------- the sampler

class NestedSampler:
    """Bayesian fit of a Pandora model to a light curve (nested sampling with pandoramoon.nss).

    likelihood: a pandoramoon.Likelihood (data, errors, supersampling, Hill-sphere threshold, ...).
    priors: dict parameter -> Uniform / LogUniform / Normal prior, or a number (fixed). Parameters: those of
        Pandora's model (PARAMS; limb darkening as q1, q2 or u1, u2). w_bary, ecc_bary, ecc_moon, w_moon
        default to 0. For model="planet" the moon parameters (and R_star, M_planet, which act only through the
        moon's orbit) are not needed.
    model: "moon" (planet + moon; also fits the planet-only model unless compare_planet=False) or "planet".
    moon_phase: "auto" (default: phase-at-transit coordinates if the priors allow), True (required), False.
    ld_correlation: correlation of Normal priors on u1 and u2 (e.g. from stellar models).
    log_dir: output folder (None: no files). resume: "subfolder" (new run<k> folder each time, as UltraNest),
        "resume" (continue / reuse a stored fit), "overwrite".
    threads: parallel runs (default: all cores available to the process). seed: random seed.
    repartition (Round 21, NEW_LEADS L1; opt-in): posterior repartitioning of 1-D priors. None (default); a dict
        {name: (mu, sigma)}: the prior of each named parameter is sampled from the mixture (1 - w) prior + w N(mu, sigma)
        (truncated to the prior) and the likelihood is weighted by prior / mixture, in both models -- Z and the
        posterior are unchanged exactly, the run is shorter if the mixture is closer to the posterior; or "auto": the
        moon model's planet parameters get mu, sigma from the planet-only fit, sigma widened by repartition_widen
        (default x20 for per_bary and t0_bary_offset, which massive moons shift, x4 otherwise).
    repartition_weight: w, the weight of the proposal N(mu, sigma) in the mixture (default 0.8): the original prior
        keeps weight 1 - w, so a misplaced proposal costs efficiency (at most ln(1 / (1 - w)) nats of H per
        parameter), never correctness. Results carry results["repartition"].
    """

    def __init__(self, likelihood, priors, model="moon", compare_planet=True, moon_phase="auto", ld_correlation=0.0,
                 log_dir=None, resume="subfolder", threads=None, seed=1, verbose=True, repartition=None,
                 repartition_weight=0.8, repartition_widen=None):
        if model not in ("moon", "planet"):
            raise ValueError('model must be "moon" or "planet"')
        if resume not in ("subfolder", "resume", "overwrite"):
            raise ValueError('resume must be "subfolder", "resume" or "overwrite"')
        unknown = set(priors) - set(PARAMS) - {"q1", "q2"}
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)} (known: {PARAMS} and q1, q2)")
        self.likelihood = likelihood
        self.priors = dict(priors)
        self.model = model
        self.verbose = verbose
        self.seed = int(seed)
        if threads is None:
            try:
                threads = len(os.sched_getaffinity(0))
            except AttributeError:
                threads = os.cpu_count() or 1
        self.threads = int(threads)
        self.models = {}
        if repartition is not None and repartition != "auto" and not isinstance(repartition, dict):
            raise ValueError('repartition must be None, "auto" or a dict {name: (mu, sigma)}')
        if repartition == "auto" and not (model == "moon" and compare_planet):
            raise ValueError('repartition="auto" needs model="moon" with the planet comparison')
        self.repartition = repartition
        self.repartition_weight = float(repartition_weight)
        self.repartition_widen = dict(per_bary=20.0, t0_bary_offset=20.0)
        self.repartition_widen.update(repartition_widen or {})
        self._moon_phase = moon_phase
        rp = repartition if isinstance(repartition, dict) else None
        if model == "moon":
            self.models["moon"] = _Model(likelihood, self.priors, "moon", moon_phase, ld_correlation, rp,
                                         self.repartition_weight)
        if model == "planet" or compare_planet:
            self.models["planet"] = _Model(likelihood, self.priors, "planet", False, ld_correlation, rp,
                                           self.repartition_weight)
        self.ld_correlation = float(ld_correlation)
        self.log_dir = None
        self.resume = resume
        if log_dir is not None:
            if resume == "subfolder":
                k = 1
                while os.path.exists(os.path.join(log_dir, f"run{k}")):
                    k += 1
                log_dir = os.path.join(log_dir, f"run{k}")
            elif resume == "overwrite" and os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
        self.results = None
        self._check_priors()

    def _check_priors(self):
        """sanity check: the prior must not be mostly outside Pandora's valid parameter range"""
        rng = np.random.default_rng(0)
        for name, m in self.models.items():
            U = rng.uniform(size=(2000, m.ndim))
            th, lw = m.native(U)
            ok = lw > LOGZERO
            from pandoramoon.pandora import params_valid
            tail = self.likelihood._tail
            valid = np.array([ok[i] and params_valid(*th[i], tail[0], tail[1], tail[4]) for i in range(len(th))])
            if valid.mean() < 0.5:
                raise ValueError(f"{name} model: only {100 * valid.mean():.0f} % of the prior is valid for Pandora "
                                 "(check the prior ranges, e.g. positive radii, periods and masses)")

    def _auto_repartition(self, planet_result):
        """repartition="auto": proposals for the moon model's planet parameters from the planet-only posterior"""
        post = planet_result["posterior"]
        spec = {}
        for j, n in enumerate(planet_result["paramnames"]):
            if n in self.models["moon"].free:
                f = float(self.repartition_widen.get(n, 4.0))
                spec[n] = (float(post["mean"][j]), f * max(float(post["stdev"][j]), 1e-300))
        self.models["moon"] = _Model(self.likelihood, self.priors, "moon", self._moon_phase, self.ld_correlation, spec,
                                     self.repartition_weight)

    def _settings_hash(self, name, nss_kw):
        m = self.models[name]
        h = hashlib.sha1()
        h.update(_data_hash(self.likelihood).encode())
        h.update(repr(sorted((k, repr(v)) for k, v in m.spec.items())).encode())
        h.update(repr((name, m.moon_phase, self.ld_correlation, self.seed, sorted(nss_kw.items()))).encode())
        h.update(repr(("nss-init", nss.INIT_VERSION)).encode())   # runs stored before Round 20 are not mixed in
        if getattr(m, "repartition", None):
            h.update(repr(sorted(m.repartition.items())).encode())
        return h.hexdigest()

    def run(self, accuracy="high", max_weighted_samples=None, **nss_kw):
        """run the fit(s). accuracy: "high" (logZ +-0.1, default) or "medium" (+-0.2, faster); max_weighted_samples:
        thin results["weighted_samples"] (all points of the merged runs, up to ~1e6) to about this many points
        (unbiased: Horvitz-Thompson thinning; None = keep all); further keyword arguments go to
        pandoramoon.nss.fit (e.g. nlive, lz_tol, ks_tol, max_runs, nsteps).
        Returns the results dict (also self.results)."""
        kw = dict(accuracy=accuracy, threads=self.threads, seed=self.seed)
        kw.update(nss_kw)
        out = {}
        order = [n for n in ("planet", "moon") if n in self.models]
        for name in order:
            m = self.models[name]
            path = None
            if self.log_dir is not None:
                path = self.log_dir if (name == self.model) else os.path.join(self.log_dir, name)
            store = _Store(path) if path else None
            if self.verbose:
                print(f"NestedSampler: {name} model, {m.ndim} sampled coordinates "
                      f"({', '.join(m.cube_names)})", flush=True)
            if name == "moon" and self.repartition == "auto":
                self._auto_repartition(out["planet"])
                m = self.models["moon"]
                if self.verbose:
                    print("  repartition (auto, weight %.2f): " % self.repartition_weight + ", ".join(
                        f"{k} N({v[0]:.6g}, {v[1]:.3g})" for k, v in m.repartition.items()), flush=True)
            mode = self.resume if self.resume == "resume" else "fresh"
            hs = self._settings_hash(name, {k: v for k, v in kw.items() if k != "threads"})
            out[name] = _fit_model(m, kw, store, mode, hs, self.verbose, name, max_weighted=max_weighted_samples)
            out[name]["repartition"] = dict(getattr(m, "repartition", {}) or {})
        res = out[self.model]
        if self.model == "moon" and "planet" in out:
            pl = out["planet"]
            m = self.models["moon"]
            if m.F_excluded > 0:
                # the excluded short-period part of the per_moon prior (all moons unphysical: planet-only
                # likelihood) contributes F * Z_planet exactly
                lz = float(np.logaddexp(res["logz"], math.log(m.F_excluded) + pl["logz"]))
                res["logz_excluded_correction"] = lz - res["logz"]
                res["logz"] = lz
            res["planet"] = pl
            res["dlogz"] = res["logz"] - pl["logz"]
            res["dlogzerr"] = float(math.hypot(res["logzerr"], pl["logzerr"]))
        elif self.model == "moon" and self.models["moon"].F_excluded > 0:
            res["logz_excluded_fraction"] = self.models["moon"].F_excluded
        if self.log_dir is not None:
            pr = {k: (v.as_dict() if isinstance(v, Prior) else v) for k, v in self.priors.items()}
            for name in order:
                path = self.log_dir if name == self.model else os.path.join(self.log_dir, name)
                _write_results(path, out[name], pr)
        self.results = res
        return res

    def print_results(self):
        """summary like UltraNest's print_results"""
        r = self.results
        if r is None:
            print("no results yet: call run()")
            return

        def block(r, title):
            print(f"{title}: logZ = {r['logz']:.3f} +- {r['logzerr']:.3f}   "
                  f"({'checks passed' if r['passed'] else 'checks NOT passed'}; {r['runs']} runs x {r['nlive']} live, "
                  f"nsteps {r['nsteps']}, {r['ncall'] / 1e6:.0f} M calls, {r['wall']:.0f} s)")
            if r.get("multimodal"):
                print(f"    WARNING: the runs disagree like runs that miss a mode (n_live escalated "
                      f"{r.get('n_nlive_escalations', 0)}x); logZ may be too low. Check with a larger nlive / max_nlive.")
            p = r["posterior"]
            for j, n in enumerate(r["paramnames"]):
                print(f"    {n:16s}: {p['median'][j]:.6g} -{p['median'][j] - p['errlo'][j]:.3g} "
                      f"+{p['errup'][j] - p['median'][j]:.3g}   (mean {p['mean'][j]:.6g}, sd {p['stdev'][j]:.3g})")
        block(r, "planet + moon" if self.model == "moon" else "planet")
        if "planet" in r:
            block(r["planet"], "planet only")
            print(f"Bayes factor: dlogZ = logZ(moon) - logZ(planet) = {r['dlogz']:.3f} +- {r['dlogzerr']:.3f}")

    def parameters(self, point=None, model=None):
        """all 20 Pandora model parameters (dict, Pandora's names; limb darkening as u1, u2) for a point of
        the free parameters (default: the maximum-likelihood point of the fit). model: None (the fitted
        model) or "planet" (the planet-only fit; its moon parameters are Pandora's planet-only values).
        Use e.g. `for k, v in sampler.parameters().items(): setattr(params, k, v)` to plot the best fit."""
        name = model or self.model
        m = self.models[name]
        if point is None:
            r = self.results if name == self.model else self.results["planet"]
            point = r["maxlikelihood"]["point"]
        point = list(point)
        vals = {}
        for k, v in m.spec.items():
            vals[k] = None if isinstance(v, Prior) else float(v)
        for k, x in zip(m.free, point):
            vals[k] = float(x)
        if "q1" in vals:
            sq = math.sqrt(vals.pop("q1"))
            q2 = vals.pop("q2")
            vals["u1"], vals["u2"] = 2.0 * sq * q2, sq * (1.0 - 2.0 * q2)
        if name == "planet":
            for k, x in zip(MOON_NAMES, (0.0, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
                vals[k] = x
        return {k: vals[k] for k in PARAMS}

    def plot_run(self, model=None, filename=None):
        """diagnostic plots of the fit (like UltraNest's run plots), saved to log_dir/plots/run.pdf:
        (1) ln Z of every merged run with the merged ln Z +- error; (2) the history of the checks: ln Z +- error and
        the chain length vs the number of runs; (3) the posterior weight of each merged point vs the posterior mass at
        higher likelihood (a well-resolved posterior: a smooth curve without isolated heavy points)."""
        import matplotlib.pyplot as plt
        r = self.results if (model is None or model == self.model) else self.results[model]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        lz = np.array(r["run_logz"])
        ax[0].errorbar(np.arange(len(lz)), lz, yerr=r["run_logzerr"], fmt="o", ms=3, color="k", alpha=0.6)
        ax[0].axhspan(r["logz"] - r["logzerr"], r["logz"] + r["logzerr"], color="C1", alpha=0.4)
        ax[0].axhline(r["logz"], color="C1")
        ax[0].set_xlabel("run (seed order)")
        ax[0].set_ylabel("ln Z")
        ax[0].set_title(f"runs: ln Z = {r['logz']:.2f} +- {r['logzerr']:.2f}")
        h = r["history"]
        x = [q["runs"] for q in h]
        ax[1].errorbar(x, [q["logz"] for q in h], yerr=[q["logz_err"] for q in h], fmt="o-", ms=3)
        ax[1].set_xlabel("runs in the result arm")
        ax[1].set_ylabel("merged ln Z")
        a2 = ax[1].twinx()
        a2.step(x, [q["nsteps"] for q in h], where="post", color="C2")
        a2.set_ylabel("nsteps", color="C2")
        ax[1].set_title("checks: " + ("passed" if r["passed"] else "NOT passed"))
        ws = r["weighted_samples"]
        o = np.argsort(ws["logl"])
        cw = np.cumsum(ws["weights"][o][::-1])[::-1]       # posterior mass above each likelihood level
        ax[2].plot(np.log(np.maximum(cw, 1e-300)), np.log(np.maximum(ws["weights"][o], 1e-300)), ".", ms=1)
        ax[2].set_xlabel("ln (posterior mass with higher L)")
        ax[2].set_ylabel("ln (posterior weight per point)")
        ax[2].set_title("weights of the merged points")
        fig.tight_layout()
        if filename is None and self.log_dir is not None:
            sub = "" if (model is None or model == self.model) else model
            filename = os.path.join(self.log_dir, sub, "plots", "run.pdf")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        if filename:
            fig.savefig(filename, bbox_inches="tight")
        return fig

    def plot_corner(self, model=None, filename=None, **kwargs):
        """corner plot of the posterior (needs the `corner` package); saved to log_dir/plots/corner.pdf"""
        import corner
        r = self.results if (model is None or model == self.model) else self.results[model]
        kwargs.setdefault("title_fmt", ".3g")      # values like R_star [m] or masses [kg] need significant digits
        fig = corner.corner(r["samples"], labels=r["paramnames"], show_titles=True, **kwargs)
        if filename is None and self.log_dir is not None:
            sub = "" if (model is None or model == self.model) else model
            filename = os.path.join(self.log_dir, sub, "plots", "corner.pdf")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        if filename:
            fig.savefig(filename, bbox_inches="tight")
        return fig                                # left open, so that notebooks display it
