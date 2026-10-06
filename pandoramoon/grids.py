import numpy as np
from numpy import sqrt, pi, cos, arcsin, empty
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def timegrid(t0_bary, epochs, epoch_duration, cadences_per_day, epoch_distance, supersampling_factor):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_bary`, however, is a free parameter
    start = np.float64(t0_bary)
    step = np.float64(epoch_distance)
    stop = np.float64(t0_bary + epoch_distance * epochs)
    ti_epoch_midtimes = np.arange(start, stop, step)

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        raise ValueError("supersampling_factor must be a positive integer")
    # Cadences are spaced by exactly 1 / cadences_per_day. Each cadence is an
    # exposure of that length, split into supersampling_factor sub-exposures
    # whose mid-times are returned (their mean is the cadence mid-time). The
    # int(...) cadences fill at most epoch_duration; the block is centred on
    # the epoch mid-time, so for integer cadences_per_day * epoch_duration
    # the exposures tile [t_start, t_end] exactly.
    # The tolerance keeps float round-off (e.g. 720 * 0.35 = 251.99999999999997)
    # from dropping a cadence; double round-off is ~1e-16 relative.
    n_float = np.float64(cadences_per_day) * np.float64(epoch_duration)
    cadences = int(n_float + 1e-9 * max(1.0, n_float)) * int(supersampling_factor)
    substep = 1.0 / (np.float64(cadences_per_day) * int(supersampling_factor))
    offsets = (np.arange(cadences) + 0.5 - cadences / 2) * substep
    time = np.empty(shape=(epochs, cadences))
    for epoch in range(epochs):
        time[epoch] = ti_epoch_midtimes[epoch] + offsets
    return time.ravel()


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_grid(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary, w_bary
):
    # Planetary transit duration at b=0 equals the width of the star
    # Formally correct would be: (R_star+r_planet) for the transit duration T1-T4
    # Here, however, we need points where center of planet is on stellar limb
    # https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    tdur = per_bary / pi * arcsin(1 / a_bary)

    # Correct transit duration based on relative orbital velocity
    # of circular versus eccentric case
    # Subtract (w_bary - 90) to match batman and PyTransit coordinate system
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))

    # t0_bary_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    x_bary = np.empty(len(time))
    for idx in range(len(time)):
        epoch = int(np.floor((time[idx] - t0_bary) / per_bary + 0.5))
        x_bary[idx] = (
            (2 * (time[idx] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
    return x_bary


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_window(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """Like x_bary_grid(), but returns only the samples with |x_bary| < reach:
    (indices into time, x_bary at those indices). NaN is kept. No full-length
    array is written, which matters for long time series (memory traffic).
    Keep the formulas identical to x_bary_grid()."""
    tdur = per_bary / pi * arcsin(1 / a_bary)
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    n = len(time)
    idx = np.empty(n, dtype=np.int64)
    x_win = np.empty(n)
    m = 0
    for i in range(n):
        epoch = int(np.floor((time[i] - t0_bary) / per_bary + 0.5))
        x = (
            (2 * (time[i] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
        if not (abs(x) >= reach):
            idx[m] = i
            x_win[m] = x
            m += 1
    return idx[:m].copy(), x_win[:m].copy()


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_window_sorted(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """Same result as x_bary_window() for ascending time, but only visits
    samples near each transit: per epoch, x_bary is linear in time,
    x = 2 (t - t_mid) / tdur with t_mid = t0_bary + per_bary * epoch +
    t0_bary_offset, so |x| < reach only for |t - t_mid| < reach * tdur / 2.
    That interval (plus a margin) is found by binary search; x and the test
    are then evaluated exactly as in x_bary_window(). Falls back to
    x_bary_window() if the window is not well inside one epoch (huge moon
    orbits, very short periods) or for non-finite constants.
    time MUST be ascending (not checked here; Likelihood checks it once)."""
    tdur = per_bary / pi * arcsin(1 / a_bary)
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    n = len(time)
    half = reach * tdur / 2  # [days]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    # Margin covers round-off of t_mid and of x (|x| errors ~1e-12)
    pad = 1e-6 * tdur + 1e-9 * (abs(time[0]) + abs(time[n - 1]))
    if not (half + abs(t0_bary_offset) + pad < 0.45 * per_bary):
        return x_bary_window(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    e_first = int(np.floor((time[0] - t0_bary) / per_bary + 0.5))
    e_last = int(np.floor((time[n - 1] - t0_bary) / per_bary + 0.5))
    if e_last - e_first + 1 > n:  # more epochs than samples: scan instead
        return x_bary_window(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    # Count first (the windows are small), then fill
    total = 0
    for e in range(e_first, e_last + 1):
        t_mid = t0_bary + per_bary * e + t0_bary_offset
        lo = np.searchsorted(time, t_mid - half - pad, side="left")
        hi = np.searchsorted(time, t_mid + half + pad, side="right")
        total += hi - lo
    idx = np.empty(total, dtype=np.int64)
    x_win = np.empty(total)
    m = 0
    for e in range(e_first, e_last + 1):
        t_mid = t0_bary + per_bary * e + t0_bary_offset
        lo = np.searchsorted(time, t_mid - half - pad, side="left")
        hi = np.searchsorted(time, t_mid + half + pad, side="right")
        for i in range(lo, hi):
            epoch = int(np.floor((time[i] - t0_bary) / per_bary + 0.5))
            x = (
                (2 * (time[i] - (t0_bary + epoch_distance * epoch))) / tdur
                - t0_shift_planet
                - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
            )
            if not (abs(x) >= reach):
                idx[m] = i
                x_win[m] = x
                m += 1
    return idx[:m].copy(), x_win[:m].copy()
