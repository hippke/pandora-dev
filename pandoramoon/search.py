"""Exomoon search with Pandora: preparation of a light curve, default priors, detection (first look) and the
epoch jackknife. The characterisation is ``pandoramoon.NestedSampler``.

    import pandoramoon as pandora
    lc = pandora.prepare_lightcurve(time, flux, t0=..., period=..., duration=...)       # needs wotan
    like = pandora.Likelihood(lc["time"], lc["flux"], lc["flux_err"], epoch_distance=period)
    priors = pandora.default_priors(t0_bary=..., per_bary=period, r_planet=0.1)
    det = pandora.detect(like, priors)            # seconds on one core: det["dlnz"], det["candidate"]
    res = pandora.NestedSampler(like, priors).run()   # minutes to hours on all cores (characterisation)

Background and the calibration of the defaults: SAMPLING.md (Rounds 11-21) and DETRENDING_STUDY.md in the
development repository; docs/Fitting.rst.
"""
import math
import time as _time
import warnings

import numpy as np

from pandoramoon import nss
from pandoramoon.likelihood import Likelihood
from pandoramoon.pandora import pandora_flux
from pandoramoon.sampling import LOGZERO, LogUniform, Normal, Prior, Uniform, _loglike, _Model  # noqa: F401

R_SUN = 696_342_000.0
M_SUN = 1.98847e30
M_JUP = 1.89813e27
G = 6.674e-11
DAY = 86400.0

# first-look presets (SAMPLING.md Round 12, examples/firstlook/README.md): target recall of well-detectable moons
# (reference dlnZ >= 8) -> planet run, moon runs; repeated small moon runs are cheaper than one large run
_PLANET = dict(nlive_planet=50, nsteps_planet=12)
RECALL_PRESETS = {0.90: dict(_PLANET, nlive_moon=50, nsteps_moon=8, repeats=1),
                  0.95: dict(_PLANET, nlive_moon=50, nsteps_moon=8, repeats=2),
                  0.99: dict(_PLANET, nlive_moon=50, nsteps_moon=16, repeats=3),
                  0.999: dict(_PLANET, nlive_moon=50, nsteps_moon=16, repeats=6)}
RETRY_BELOW, RETRY_MAX = -15.0, 2
# epoch jackknife: a moon must not rest on one transit. Delta chi^2 (planet-only vs planet+moon, at their best
# fits, with the likelihood's errors) without the epoch that carries most of it; >= JACK_MIN for a candidate.
# 18 ~ 2 x (threshold 3 + Occam penalty ~6). Provisional: Kepler controls (SAMPLING.md Rounds 20-21) reached 19 in
# one of 15 nulls; the full set of 45 nulls recalibrates it.
JACK_MIN = 18.0


# ------------------------------------------------------------------------------------------------ priors

def default_priors(t0_bary, per_bary, r_planet, per_bary_err=None, a_bary=None, R_star=R_SUN, M_star=M_SUN,
                   M_planet=M_JUP, t0_halfwidth=0.05, per_moon_max=None):
    """Standard priors for a moon search around a known planet (the priors of the first-look calibration,
    scaled to the planet). Returns a dict for NestedSampler / detect; change any entry as needed.

    t0_bary [d]: a reference transit time (fixed; its uncertainty is t0_bary_offset +- t0_halfwidth [d]).
    per_bary [d]: the planet period; prior +- max(5 per_bary_err, 0.002 per_bary).
    r_planet [R_star]: the planet radius estimate; prior 0.5 ... 1.5 x.
    a_bary [R_star]: (lo, hi) or None (Kepler's law with M_star + M_planet, +- 2.3 %).
    R_star [m], M_star [kg], M_planet [kg]: estimates; priors R_star 0.8 ... 1.2 x, M_planet 0.5 ... 1.5 x.
    Moon: r_moon 0.1 ... 0.7 r_planet, per_moon 0 ... per_moon_max (default 0.19 per_bary: the period at the
    planet's Hill radius), tau 0 ... 1, Omega 0 ... 180 deg, i 0 ... 180 deg, M_moon 0 ... 0.5 M_planet.

    The Bayes factor depends on the moon priors (their volume sets the Occam penalty); the detection threshold of
    detect() was calibrated with priors of this kind."""
    P = float(per_bary)
    dP = max(5.0 * float(per_bary_err or 0.0), 0.002 * P)
    if a_bary is None:
        a = (G * (M_star + M_planet) * (P * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_star
        a_bary = (0.977 * a, 1.023 * a)
    k = float(r_planet)
    pm_max = float(per_moon_max) if per_moon_max is not None else round(0.19 * P, 3)
    return dict(
        q1=Uniform(0, 1), q2=Uniform(0, 1),
        R_star=Uniform(0.8 * R_star, 1.2 * R_star),
        per_bary=Uniform(P - dP, P + dP),
        a_bary=Uniform(*a_bary),
        r_planet=Uniform(0.5 * k, 1.5 * k),
        b_bary=Uniform(0, 1),
        t0_bary=float(t0_bary),
        t0_bary_offset=Uniform(-t0_halfwidth, t0_halfwidth),
        M_planet=Uniform(0.5 * M_planet, 1.5 * M_planet),
        r_moon=Uniform(0.1 * k, 0.7 * k),
        per_moon=Uniform(0, pm_max),
        tau_moon=Uniform(0, 1),
        Omega_moon=Uniform(0, 180),
        i_moon=Uniform(0, 180),
        M_moon=Uniform(0, 0.5 * M_planet),
    )


# --------------------------------------------------------------------------------------- light curve

def subexposure_times(time, factor, exposure=None):
    """sub-exposure mid-times (cadence-major, factor per cadence) as Pandora's supersampled grids; exposure [d]
    defaults to the cadence (median time step)"""
    t = np.asarray(time, dtype=float)
    if exposure is None:
        exposure = cadence_of(t)
    off = (np.arange(factor) + 0.5 - factor / 2) * exposure / factor
    return (t[:, None] + off[None, :]).ravel()


def cadence_of(time):
    """the cadence [d]: the median of the positive time steps (gaps between transit windows do not matter)"""
    d = np.diff(np.asarray(time, dtype=float))
    d = d[d > 0]
    if len(d) == 0:
        raise ValueError("cannot determine the cadence from fewer than 2 distinct times")
    return float(np.median(d))


def resample_likelihood(like, factor, exposure=None):
    """a copy of a Likelihood with factor sub-exposures per cadence (the original must have factor 1)"""
    if like.supersampling_factor == factor:
        return like
    if like.supersampling_factor != 1:
        raise ValueError("resample_likelihood: the likelihood is already supersampled")
    err = np.where(like.inv_err > 0, 1.0 / np.where(like.inv_err > 0, like.inv_err, 1.0), np.nan)
    data = np.where(like.inv_err > 0, like.data, np.nan)
    return Likelihood(subexposure_times(like.time, factor, exposure), data, err, epoch_distance=like.epoch_distance,
                      supersampling_factor=factor, occult_small_threshold=like.occult_small_threshold,
                      hill_sphere_threshold=like.hill_sphere_threshold, numerical_grid=like.numerical_grid)


def cadence_times(like):
    """the output-cadence times of a Likelihood (mean of each cadence's sub-exposures)"""
    k = like.supersampling_factor
    if k == 1:
        return like.time
    n = len(like.data)
    return like.time[: n * k].reshape(n, k).mean(axis=1)


def prepare_lightcurve(time, flux, t0, period, duration, mask=1.0, window=4.0, method="biweight",
                       fit_halfwidth=None, inflate_beta=True):
    """Detrend a light curve around known transits and estimate its noise (needs wotan, ``pip install wotan``).

    time [d], flux: the light curve (e.g. Kepler PDCSAP, normalised or not). t0 [d], period [d], duration [d]:
    the planet's ephemeris and transit duration T14 (e.g. from TLS / BLS).
    mask, window: the transits are masked over mask x T14 and wotan's `method` (default biweight) is fitted with a
    window of window x T14. Defaults 1.0 / 4.0: the best setting of the detrending study (300 injected planet+moon
    systems in 10 quiet Kepler stars; robust to T14 errors of -20 ... +25 %). Points where wotan has no trend (e.g.
    near gaps) are dropped, not interpolated.
    fit_halfwidth [d]: the fit windows are t0 + k period +- fit_halfwidth (default (mask / 2 + 1) T14); increase it
    for moons that may transit further from the planet.
    Noise: sigma = the robust scatter (MAD) of the detrended flux 1 ... 11 T14 beyond the fit windows (signal-free),
    beta = the red-noise factor of the same points on bins of T14 / 4 (Pont, Zucker & Queloz 2006), >= 1.
    flux_err = sigma x beta if inflate_beta else sigma.

    Returns a dict: time, flux, flux_err (the fit windows, for Likelihood), sigma, beta, epochs (transit numbers
    with data), and for plots: time_all, flux_all (input), trend, flat (detrended), in_window (boolean)."""
    try:
        from wotan import flatten
    except ImportError:
        raise ImportError("prepare_lightcurve needs wotan: pip install wotan")
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    o = np.argsort(t)
    t, f = t[o], f[o]
    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    msk = np.abs(ph) < 0.5 * mask * duration
    kw = dict(method=method, window_length=window * duration, return_trend=True, mask=msk, break_tolerance=0.5)
    if method == "cosine":
        kw["robust"] = True
    flat, trend = flatten(t, f, **kw)
    half = float(fit_halfwidth) if fit_halfwidth is not None else (0.5 * mask + 1.0) * duration
    win = (np.abs(ph) < half) & np.isfinite(flat)
    far = np.isfinite(flat) & (np.abs(ph) > half + duration) & (np.abs(ph) < half + 11 * duration)
    if far.sum() < 50:
        far = np.isfinite(flat) & (np.abs(ph) > half)
    r = flat[far] - np.median(flat[far])
    sig = 1.4826 * float(np.median(np.abs(r)))
    beta = 1.0
    key = np.floor((t[far] - t[far].min()) / (duration / 4)).astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mb = cnt >= 3
    if mb.sum() > 10:
        binned = np.bincount(inv, weights=r)[mb] / cnt[mb]
        beta = max(1.0, float(np.sqrt(np.mean(binned ** 2 / (sig ** 2 / cnt[mb])))))
    err = sig * beta if inflate_beta else sig
    epochs = np.unique(np.round((t[win] - t0) / period).astype(int))
    return dict(time=t[win], flux=flat[win], flux_err=np.full(int(win.sum()), err), sigma=sig, beta=beta,
                epochs=epochs.tolist(), time_all=t, flux_all=f, trend=trend, flat=flat, in_window=win,
                settings=dict(mask=mask, window=window, method=method, fit_halfwidth=half, duration=duration,
                              inflate_beta=inflate_beta))


# ---------------------------------------------------------------------------------------- jackknife

def model_flux(like, params):
    """Pandora's total flux for a parameter dict (20 parameters, e.g. NestedSampler.parameters()) at the
    likelihood's output cadence"""
    p = params
    _, _, f = pandora_flux(p["u1"], p["u2"], p["R_star"], p["per_bary"], p["a_bary"], p["r_planet"], p["b_bary"],
                           p["w_bary"], p["ecc_bary"], p["t0_bary"], p["t0_bary_offset"], p["M_planet"], p["r_moon"],
                           p["per_moon"], p["tau_moon"], p["Omega_moon"], p["i_moon"], p["ecc_moon"], p["w_moon"],
                           p["M_moon"], like.epoch_distance, like.supersampling_factor, like.occult_small_threshold,
                           like.hill_sphere_threshold, like.numerical_grid, like.time)
    return np.asarray(f)


def epoch_jackknife(like, params_moon, params_planet, jack_min=JACK_MIN):
    """How the improvement of the planet+moon fit over the planet-only fit is spread over the transits.
    Per transit epoch: Delta chi^2 = chi^2(planet only) - chi^2(planet + moon) at the two best fits (with the
    likelihood's errors). A moon seen in several transits passes; a moon model that only fits a feature of one
    transit (e.g. an instrumental or stellar artefact) does not.
    Returns dict: epochs, dchi2_epochs, dchi2, top_epoch, top_share, dchi2_without_top, passed
    (dchi2_without_top >= jack_min and >= 2 epochs), jack_min."""
    tc = cadence_times(like)
    fm, fp = model_flux(like, params_moon), model_flux(like, params_planet)
    w = like.inv_var
    t0 = params_planet["t0_bary"] + params_planet["t0_bary_offset"]
    ep = np.round((tc - t0) / params_planet["per_bary"]).astype(int)
    d = like.data
    epochs = np.unique(ep[w > 0])
    dchi = np.array([float(np.sum(w[ep == e] * ((d[ep == e] - fp[ep == e]) ** 2 - (d[ep == e] - fm[ep == e]) ** 2)))
                     for e in epochs])
    tot = float(dchi.sum()) if len(dchi) else 0.0
    j = int(np.argmax(dchi)) if len(dchi) else -1
    wo = tot - float(dchi[j]) if len(dchi) else 0.0
    return dict(epochs=[int(e) for e in epochs], dchi2_epochs=[float(x) for x in dchi], dchi2=tot,
                top_epoch=int(epochs[j]) if j >= 0 else None,
                top_share=float(dchi[j] / tot) if tot > 0 else None, dchi2_without_top=wo,
                passed=bool(len(epochs) >= 2 and wo >= jack_min), jack_min=float(jack_min))


def _best_params(m, U, L):
    """parameter dict (20) of the cube point with the highest likelihood (without the coordinate weight)"""
    th, lw = m.native(U)
    ok = lw > LOGZERO
    i = int(np.argmax(np.where(ok, L - lw, -np.inf)))
    from pandoramoon.sampling import PARAMS
    return {n: float(th[i, k]) for k, n in enumerate(PARAMS)}


# ----------------------------------------------------------------------------------------- detection

def detect(likelihood, priors, recall=0.95, threshold=3.0, seed=1, jack_min=JACK_MIN, ld_correlation=0.0,
           verbose=False):
    """Fast moon detection test ("first look"; one core, seconds per light curve).

    One short nested-sampling run of the planet-only model and a few short runs of the planet+moon model (same
    priors for the shared parameters), giving the Bayes factor dlnz = ln Z(planet+moon) - ln Z(planet only), and the
    epoch jackknife of the two best fits.
    recall: 0.90, 0.95 (default), 0.99 or 0.999: the fraction of well-detectable moons (full-fit dlnZ >= 8) that
    the settings are calibrated to find (more recall = more moon runs = more CPU: 0.35 / 0.6 / 1.6 / 3.1 CPU-s per
    light curve in the calibration).
    threshold: dlnz above which the light curve is "interesting" (3: calibrated on white-noise injections with
    priors like default_priors(): 0 false passes in 720 no-moon light curves at recall 0.95; for real light curves
    use errors with the red-noise factor, see prepare_lightcurve).
    The evidences of a first look are noisy (+-0.5) and slightly biased; a candidate must be confirmed with the full
    fit (NestedSampler).

    Returns a dict: dlnz, dlnz_err, logz_planet, logz_moon, interesting (dlnz > threshold), jackknife (see
    epoch_jackknife), candidate (interesting and the jackknife passed), moon (median and 16 / 84 % of per_moon,
    r_moon, M_moon / M_planet, i_moon from the best moon run), best (best-fit parameters of both models), warnings,
    calls, cpu seconds, settings."""
    if recall not in RECALL_PRESETS:
        raise ValueError(f"recall must be one of {sorted(RECALL_PRESETS)}")
    st = dict(RECALL_PRESETS[recall], threshold=float(threshold))
    mm = _Model(likelihood, priors, "moon", "auto", ld_correlation)
    mp = _Model(likelihood, priors, "planet", False, ld_correlation)
    c0 = _time.process_time()
    w0 = _time.perf_counter()

    def run(m, nlive, nsteps, sd):
        return nss.run(_loglike, m.ndim, wrapped=m.wrapped, nlive=nlive, nsteps=nsteps, seed=sd, dirs="pairs",
                       kpair=4, data=m.D)
    rp = run(mp, st["nlive_planet"], st["nsteps_planet"], seed)
    rms = [run(mm, st["nlive_moon"], st["nsteps_moon"], seed + 7 + 101 * j) for j in range(st["repeats"])]

    def lz_moon(rs):
        lz = float(np.logaddexp.reduce([r["logz"] for r in rs]) - math.log(len(rs)))
        if mm.F_excluded > 0:                 # excluded short moon periods: planet-only likelihood (exact)
            lz = float(np.logaddexp(lz, math.log(mm.F_excluded) + rp["logz"]))   # as NestedSampler.run
        return lz
    lzm = lz_moon(rms)
    nretry = 0
    while lzm - rp["logz"] < RETRY_BELOW and nretry < RETRY_MAX:
        # the moon model (almost) contains the planet-only model: a moon run far below it has collapsed
        nretry += 1
        rms.append(run(mm, st["nlive_moon"], st["nsteps_moon"], seed + 7 + 101 * (st["repeats"] + nretry)))
        good = [r for r in rms if r["logz"] - rp["logz"] >= RETRY_BELOW] or rms
        lzm = lz_moon(good)
    rm = max(rms, key=lambda r: r["logz"])
    dl = lzm - rp["logz"]
    bm = _best_params(mm, rm["dead_u"], rm["dead_L"])
    bp = _best_params(mp, rp["dead_u"], rp["dead_L"])
    jk = epoch_jackknife(likelihood, bm, bp, jack_min)
    pars = mm.to_params(rm["samples_u"])
    names = mm.paramnames()
    col = {n: pars[:, i] for i, n in enumerate(names)}
    summ = {}
    for n in ("per_moon", "r_moon", "i_moon"):
        if n in col:
            summ[n] = tuple(float(v) for v in np.percentile(col[n], [50, 16, 84]))
    if "M_moon" in col and "M_planet" in col:
        summ["M_moon/M_planet"] = tuple(float(v) for v in np.percentile(col["M_moon"] / col["M_planet"], [50, 16, 84]))
    interesting = bool(dl > st["threshold"])
    warn = []
    if len(jk["epochs"]) < 2:
        warn.append("fewer than 2 transits with data: the moon signal cannot be checked across transits")
    if interesting and not jk["passed"]:
        warn.append(f"dlnz {dl:.1f} > threshold, but {100 * (jk['top_share'] or 1):.0f} % of the fit improvement "
                    f"comes from one transit (epoch {jk['top_epoch']}): likely an artefact of that transit, "
                    "not a moon")
    out = dict(dlnz=float(dl), dlnz_err=float(math.hypot(rm["logzerr"], rp["logzerr"])), logz_planet=rp["logz"],
               logz_moon=lzm, logz_moon_runs=[r["logz"] for r in rms], interesting=interesting, jackknife=jk,
               candidate=bool(interesting and jk["passed"]), moon=summ, best=dict(moon=bm, planet=bp), warnings=warn,
               calls=int(rp["ncall"] + sum(r["ncall"] for r in rms)), cpu=_time.process_time() - c0,
               wall=_time.perf_counter() - w0, retries=nretry, settings=dict(st, recall=recall, jack_min=jack_min))
    if verbose:
        print(f"detect: dlnZ = {dl:+.2f} +- {out['dlnz_err']:.2f} (threshold {threshold}); jackknife: "
              f"{jk['dchi2_without_top']:.1f} without the strongest transit (need >= {jack_min}); "
              f"candidate: {out['candidate']}" + "".join("\n  note: " + w for w in warn))
    return out
