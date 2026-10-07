import numpy as np
from numpy import sqrt, pi, sin, cos, abs
from numba import jit

from pandoramoon.trig import sincos_turns, sincos_small, SMALL_PHI_MAX, TWO_PI


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
    """2D x-y Kepler solver WITH eccentricity: (xm, ym, xp, yp). Since P7b
    the positions of ellipse_ecc_z() (vectorised solver, no true anomaly)."""
    xm, ym, xp, yp, _, _ = ellipse_ecc_z(a, per, e, tau, Omega, w, i, time, mass_ratio, x_bary, b_bary)
    return xm, ym, xp, yp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_z(a, per, tau, Omega, i, time, x_bary, mass_ratio, b_bary):
    """ellipse() fused into one loop that also returns the squared distances
    from the stellar centre: (xm, ym, xp, yp, zm2, zp2), z2 = x^2 + y^2 (P26:
    the occultations take z^2, so no sqrt here). No full-length temporaries
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
    zm2 = np.empty(n)
    zp2 = np.empty(n)
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
        zm2[j] = xmj * xmj + ymj * ymj
        zp2[j] = xpj * xpj + ypj * ypj
    return xm, ym, xp, yp, zm2, zp2


# P24d: the orbit loops on the runs evaluate the moon's phase in blocks of
# samples around an anchor sample: one sincos_turns() per block and, per
# sample, sin / cos of the angle phi = omega (t - t_anchor) from the short
# polynomial trig.sincos_small() (|phi| <= SMALL_PHI_MAX), combined by angle
# addition folded into the projection constants. Blocks are balanced,
# multiples of 4 samples (no scalar remainder except at the end of a run)
# and at most ANCHOR_BMAX long; runs whose blocks would be shorter than
# ANCHOR_BMIN samples (short moon periods at coarse cadence, ss1) use
# sincos_turns() per sample as before, so does any block whose time span
# exceeds the polynomial's range (gaps).
ANCHOR_BMIN = 32
ANCHOR_BMAX = 4096


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _anchor_block(tr, cnt, omega):
    """Block length [samples] for the run tr[:cnt] (ascending): balanced
    multiple of 4 with a half-span of about SMALL_PHI_MAX / omega on the
    mean step; cnt (one block, checked by the caller) if that is below
    ANCHOR_BMIN; >= 1"""
    if cnt < 2:
        return 1
    w = omega * ((tr[cnt - 1] - tr[0]) / (cnt - 1))
    hb = ANCHOR_BMAX if w * ANCHOR_BMAX <= 2.0 * SMALL_PHI_MAX else 2.0 * SMALL_PHI_MAX / w
    if not hb >= ANCHOR_BMIN:  # also NaN
        return cnt
    b4 = (int(hb) >> 2) << 2
    nblk = (cnt + b4 - 1) // b4
    return ((((cnt + nblk - 1) // nblk) + 3) >> 2) << 2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_z_runs(
    a, per, tau, Omega, i, time, los, his, c_t, c_d, tdur, t0_shift_planet,
    mass_ratio, b_bary
):
    """ellipse_z() on the window runs of grids.x_bary_runs_sorted() (P18):
    the samples are time[los[k]:his[k]] for all k, in this order (each run
    ascending), and x_bary is computed inline with the expression of
    grids.runs_x_t(), so no window arrays (idx, x_bary, time) are filled
    and read back. Same output as ellipse_z() on the materialised window,
    to the ulp of the phase (t - tau per) / per: the phase comes from block
    anchors plus a small-angle polynomial of t - t_anchor (P24d, see
    ANCHOR_BMIN). Returns (xm, ym, xp, yp, zm2, zp2) with the squared
    distances z2 = x^2 + y^2 (P26: no sqrt per sample)."""
    O = Omega / 180 * pi
    inc = i / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    cos_i = cos(inc)
    sc = sin_O * cos_i
    cc = cos_O * cos_i
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    n = 0
    for k in range(len(los)):
        n += his[k] - los[k]
    xm = np.empty(n)
    ym = np.empty(n)
    xp = np.empty(n)
    yp = np.empty(n)
    zm2 = np.empty(n)
    zp2 = np.empty(n)
    t_ref = tau * per
    omega = abs(TWO_PI / per)
    m = 0
    for k in range(len(los)):
        ct = c_t[k]
        cd = c_d[k]
        lo = los[k]
        cnt = his[k] - lo
        # slices: indices from range() are known non-negative, so numba
        # emits no wraparound selects and the loop vectorises (time[lo + j]
        # does not)
        tr = time[lo:lo + cnt]
        bl = _anchor_block(tr, cnt, omega)
        for b0 in range(0, cnt, bl):
            b1 = min(b0 + bl, cnt)
            nb = b1 - b0
            ta = tr[b0 + (nb >> 1)]
            trb = tr[b0:b1]
            xmr = xm[m + b0:m + b1]
            ymr = ym[m + b0:m + b1]
            xpr = xp[m + b0:m + b1]
            ypr = yp[m + b0:m + b1]
            zmr = zm2[m + b0:m + b1]
            zpr = zp2[m + b0:m + b1]
            if max(ta - trb[0], trb[nb - 1] - ta) * omega <= SMALL_PHI_MAX:
                # Q = Q_a + phi: (cos Q, sin Q) = R(phi) (cos Q_a, sin Q_a),
                # folded into the projection
                s_a, c_a = sincos_turns((ta - t_ref) / per)
                vxc = cos_O * c_a - sc * s_a
                vxs = -cos_O * s_a - sc * c_a
                vyc = sin_O * c_a + cc * s_a
                vys = cc * c_a - sin_O * s_a
                for j in range(nb):
                    t = trb[j]
                    x_bary = (2 * (t - ct)) / tdur - t0_shift_planet - cd
                    sin_p, cos_p = sincos_small((t - ta) * omega)
                    vector_x = vxc * cos_p + vxs * sin_p
                    vector_y = vyc * cos_p + vys * sin_p
                    xmj = vector_x * a_moon + x_bary
                    ymj = vector_y * a_moon + b_bary
                    xpj = -vector_x * a_planet + x_bary
                    ypj = -vector_y * a_planet + b_bary
                    xmr[j] = xmj
                    ymr[j] = ymj
                    xpr[j] = xpj
                    ypr[j] = ypj
                    zmr[j] = xmj * xmj + ymj * ymj
                    zpr[j] = xpj * xpj + ypj * ypj
            else:
                for j in range(nb):
                    t = trb[j]
                    x_bary = (2 * (t - ct)) / tdur - t0_shift_planet - cd
                    sin_Q, cos_Q = sincos_turns((t - t_ref) / per)
                    vector_x = cos_O * cos_Q - sin_O * sin_Q * cos_i
                    vector_y = sin_O * cos_Q + cos_O * sin_Q * cos_i
                    xmj = vector_x * a_moon + x_bary
                    ymj = vector_y * a_moon + b_bary
                    xpj = -vector_x * a_planet + x_bary
                    ypj = -vector_y * a_planet + b_bary
                    xmr[j] = xmj
                    ymr[j] = ymj
                    xpr[j] = xpj
                    ypr[j] = ypj
                    zmr[j] = xmj * xmj + ymj * ymj
                    zpr[j] = xpj * xpj + ypj * ypj
        m += cnt
    return xm, ym, xp, yp, zm2, zp2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_z_only_runs(
    a, per, tau, Omega, i, time, los, his, c_t, c_d, tdur, t0_shift_planet,
    mass_ratio, b_bary
):
    """ellipse_z_runs() without the positions: (zm2, zp2) only (P24b). For
    draws where no planet-moon eclipse is possible
    (eclipse.eclipse_impossible_circular), the positions are not needed;
    two output arrays instead of six (fewer stores and allocations), same
    arithmetic per sample as ellipse_z_runs() (incl. P24d's block
    anchors)."""
    O = Omega / 180 * pi
    inc = i / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    cos_i = cos(inc)
    sc = sin_O * cos_i
    cc = cos_O * cos_i
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    n = 0
    for k in range(len(los)):
        n += his[k] - los[k]
    zm2 = np.empty(n)
    zp2 = np.empty(n)
    t_ref = tau * per
    omega = abs(TWO_PI / per)
    m = 0
    for k in range(len(los)):
        ct = c_t[k]
        cd = c_d[k]
        lo = los[k]
        cnt = his[k] - lo
        tr = time[lo:lo + cnt]
        bl = _anchor_block(tr, cnt, omega)
        for b0 in range(0, cnt, bl):
            b1 = min(b0 + bl, cnt)
            nb = b1 - b0
            ta = tr[b0 + (nb >> 1)]
            trb = tr[b0:b1]
            zmr = zm2[m + b0:m + b1]
            zpr = zp2[m + b0:m + b1]
            if max(ta - trb[0], trb[nb - 1] - ta) * omega <= SMALL_PHI_MAX:
                s_a, c_a = sincos_turns((ta - t_ref) / per)
                vxc = cos_O * c_a - sc * s_a
                vxs = -cos_O * s_a - sc * c_a
                vyc = sin_O * c_a + cc * s_a
                vys = cc * c_a - sin_O * s_a
                for j in range(nb):
                    t = trb[j]
                    x_bary = (2 * (t - ct)) / tdur - t0_shift_planet - cd
                    sin_p, cos_p = sincos_small((t - ta) * omega)
                    vector_x = vxc * cos_p + vxs * sin_p
                    vector_y = vyc * cos_p + vys * sin_p
                    xmj = vector_x * a_moon + x_bary
                    ymj = vector_y * a_moon + b_bary
                    xpj = -vector_x * a_planet + x_bary
                    ypj = -vector_y * a_planet + b_bary
                    zmr[j] = xmj * xmj + ymj * ymj
                    zpr[j] = xpj * xpj + ypj * ypj
            else:
                for j in range(nb):
                    t = trb[j]
                    x_bary = (2 * (t - ct)) / tdur - t0_shift_planet - cd
                    sin_Q, cos_Q = sincos_turns((t - t_ref) / per)
                    vector_x = cos_O * cos_Q - sin_O * sin_Q * cos_i
                    vector_y = sin_O * cos_Q + cos_O * sin_Q * cos_i
                    xmj = vector_x * a_moon + x_bary
                    ymj = vector_y * a_moon + b_bary
                    xpj = -vector_x * a_planet + x_bary
                    ypj = -vector_y * a_planet + b_bary
                    zmr[j] = xmj * xmj + ymj * ymj
                    zpr[j] = xpj * xpj + ypj * ypj
        m += cnt
    return zm2, zp2


# P10: per-body windows inside the epoch runs. Moon cells span about
# sqrt(8 SUBRUN_SLACK / (a_m omega^2)) days (curvature slack SUBRUN_SLACK
# [R_star]), at most SUBRUN_DT_MAX days and SUBRUN_H_MAX samples. Runs whose
# cells would hold fewer than SUBRUN_H_MIN samples are kept whole: a cell
# point costs ~10 ns, about what the orbit loop saves on ~4-5 samples
# (Kepler long cadence without supersampling).
SUBRUN_SLACK = 0.01
SUBRUN_DT_MAX = 0.1
SUBRUN_H_MIN = 8
SUBRUN_H_MAX = 4096


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _emit_interval(s, e, cur_s, cur_e, los2, his2, n_out):
    """Append [s, e) to the merged interval list (starts non-decreasing)"""
    if cur_s < 0:
        return s, e, n_out
    if s <= cur_e:
        return cur_s, max(cur_e, e), n_out
    los2[n_out] = cur_s
    his2[n_out] = cur_e
    return s, e, n_out + 1


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _cell_samples(time, lo, hi, cell_dt):
    """Moon cell length [samples] in the run time[lo:hi] for cells of
    ~cell_dt days; 0 (keep the run whole) if that is below SUBRUN_H_MIN
    samples or the run is shorter than that"""
    cnt = hi - lo
    if cnt < SUBRUN_H_MIN:
        return 0
    dt = (time[hi - 1] - time[lo]) / (cnt - 1)
    if not dt > 0:
        return SUBRUN_H_MAX
    hf = cell_dt / dt
    return 0 if hf < SUBRUN_H_MIN else (SUBRUN_H_MAX if not hf < SUBRUN_H_MAX else int(hf))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def subrun_cell_dt(a, per, mass_ratio):
    """Moon cell length [days] of ellipse_subruns() and its curvature
    coefficient: (cell_dt, curv), curv (t_b - t_a)^2 = bound on the
    deviation of the moon's sky track from its chord over a cell"""
    a_moon = a - (a * mass_ratio) / (1 + mass_ratio)
    omega = 2 * pi / per
    curv = a_moon * omega * omega * (1 + 1e-9) / 8
    cell_dt = min(sqrt(SUBRUN_SLACK / curv), SUBRUN_DT_MAX) if curv > 0 else SUBRUN_DT_MAX
    return cell_dt, curv


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def subruns_pay(time, los, his, cell_dt):
    """True if any run has moon cells of >= SUBRUN_H_MIN samples, i.e.
    ellipse_subruns() can split it (otherwise it would return the runs
    unchanged; the caller skips it)"""
    for k in range(len(los)):
        if _cell_samples(time, los[k], his[k], cell_dt) > 0:
            return True
    return False


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _search_sorted(tr, x, right, t0, scale):
    """np.searchsorted(tr, x, side="right" if right else "left") for
    ascending tr, walked from the linear-interpolation guess (x - t0) scale
    (scale = (len - 1) / (tr[-1] - t0)): 0-2 steps on uniform sampling,
    binary search after 16 steps (gaps)"""
    n = len(tr)
    g = (x - t0) * scale
    if not g > 0:
        j = 0
    elif not g < n:
        j = n
    else:
        j = int(g)
    steps = 0
    if right:
        while j > 0 and tr[j - 1] > x:
            j -= 1
            steps += 1
            if steps > 16:
                return np.searchsorted(tr, x, side="right")
        while j < n and not tr[j] > x:
            j += 1
            steps += 1
            if steps > 16:
                return np.searchsorted(tr, x, side="right")
    else:
        while j > 0 and not tr[j - 1] < x:
            j -= 1
            steps += 1
            if steps > 16:
                return np.searchsorted(tr, x, side="left")
        while j < n and tr[j] < x:
            j += 1
            steps += 1
            if steps > 16:
                return np.searchsorted(tr, x, side="left")
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _moon_cells(
    tr, s, e, h, lo, t_ref, inv_per, cos_O, sin_O, cos_i, a_moon, ct, two_tdur,
    sh, b_bary, z_lim_m, curv, cur_s, cur_e, los2, his2, n_out
):
    """Moon cells [ja, jb] (samples s, s + h, ..., e of the run tr, e
    included; e > s) whose arc can come closer to the star centre than
    z_lim_m, appended (as lo + ja .. lo + jb + 1) to the merged interval list.
    Kept if the distance of the centre from the chord P_a P_b, minus the
    curvature bound curv (t_b - t_a)^2, is < z_lim_m; tested without
    division or sqrt: with lim = z_lim_m + curv dt^2 and s_ = -P_a.(P_b -
    P_a), the closest chord point is P_a (s_ <= 0), P_b (s_ >= |P_b -
    P_a|^2) or the foot of the perpendicular (distance |P_a x (P_b - P_a)| /
    |P_b - P_a|)."""
    ja = s
    ta = tr[ja]
    sin_Q, cos_Q = sincos_turns((ta - t_ref) * inv_per)
    xa = (cos_O * cos_Q - sin_O * sin_Q * cos_i) * a_moon + ((ta - ct) * two_tdur - sh)
    ya = (sin_O * cos_Q + cos_O * sin_Q * cos_i) * a_moon + b_bary
    while ja < e:
        jb = min(ja + h, e)
        tb = tr[jb]
        sin_Q, cos_Q = sincos_turns((tb - t_ref) * inv_per)
        xb = (cos_O * cos_Q - sin_O * sin_Q * cos_i) * a_moon + ((tb - ct) * two_tdur - sh)
        yb = (sin_O * cos_Q + cos_O * sin_Q * cos_i) * a_moon + b_bary
        dx = xb - xa
        dy = yb - ya
        dtc = tb - ta
        lim = z_lim_m + curv * dtc * dtc
        lim2 = lim * lim
        s_ = -(xa * dx + ya * dy)
        dd = dx * dx + dy * dy
        if s_ <= 0:
            keep = xa * xa + ya * ya < lim2
        elif s_ >= dd:
            keep = xb * xb + yb * yb < lim2
        else:
            cr = xa * dy - ya * dx
            keep = cr * cr < lim2 * dd
        if keep:
            cur_s, cur_e, n_out = _emit_interval(lo + ja, lo + jb + 1, cur_s, cur_e, los2, his2, n_out)
        ja = jb
        ta = tb
        xa = xb
        ya = yb
    return cur_s, cur_e, n_out


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse_subruns(
    a, per, tau, Omega, i, time, los, his, c_t, c_d, tdur, t0_shift_planet,
    mass_ratio, b_bary, r_planet, r_moon
):
    """P10: the parts of the window runs (grids.x_bary_runs_sorted) where
    the planet or the circular moon can be on the star: (los2, his2, c_t2,
    c_d2), sub-runs in the same form, ascending and disjoint. All other
    window samples have planet and moon flux exactly 1 (no occultation, and
    eclipse() only changes a moon flux < 1), so they need no orbit,
    occultation or chi^2 work. Conservative by construction:

    - Planet: offset a_p = a q / (1 + q) from the barycentre (circular), so
      |x_p| >= |x_bary| - a_p and |y_p| >= |b| - a_p; on the star
      (z_p < 1 + r_p) only if |x_bary| < a_p + sqrt((1 + r_p)^2 -
      max(|b| - a_p, 0)^2). x_bary is linear in t within a run: one
      interval [pl, ph), found by search (padded: extra samples are
      harmless).
    - Moon (P20: first the analytic range): a_m v = a_m (cos O cos Q - sin O
      sin Q cos i, sin O cos Q + cos O sin Q cos i) has |a_m v_x| <= A_x =
      a_m sqrt(cos^2 O + sin^2 O cos^2 i) and |a_m v_y| <= A_y = a_m
      sqrt(sin^2 O + cos^2 O cos^2 i), so the moon is on the star only if
      |b| - A_y < 1 + r_moon and |x_bary| < 1 + r_moon + A_x: one interval
      [ml, mh) per run, found like the planet's. Inside it, the sky
      position P(t) = (x_bary(t) + a_m v_x, b + a_m v_y) at every h-th
      sample (and the last one; a_m = a / (1 + q)). x_bary is linear in t
      and |d^2 v / dt^2| = omega^2 |v| <= omega^2, so on a cell [t_a, t_b]
      P deviates from the chord P_a P_b by at most a_m omega^2 (t_b -
      t_a)^2 / 8: the cell is kept if the distance of the star centre from
      the chord minus that is < 1 + r_moon (_moon_cells). Cells inside the
      planet interval would be kept anyway, so they are not evaluated
      (P20): cells cover ml .. min(pl, mh - 1) and max(ph - 1, ml) .. mh -
      1. h gives cells of ~sqrt(8 SUBRUN_SLACK / (a_m omega^2)) days, at
      most SUBRUN_DT_MAX (shorter cells keep fewer samples, but cost a
      point evaluation each, ~7 ns); runs with cells of fewer than
      SUBRUN_H_MIN samples are returned whole.
    - Margins 1e-6 (R_star) on all bounds cover round-off (the cell points
      use reciprocals, not the orbit loop's divisions: ~1e-12 R_star)."""
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    cell_dt, curv = subrun_cell_dt(a, per, mass_ratio)
    n_runs = len(los)
    # Output capacity: per run at most one sub-run per moon cell (cells on
    # both sides of the planet interval: one partial cell more) plus the
    # planet's
    cap = 1
    for k in range(n_runs):
        h = _cell_samples(time, los[k], his[k], cell_dt)
        cnt = his[k] - los[k]
        cap += (cnt + h - 1) // h + 2 if h > 0 else 1
    O = Omega / 180 * pi
    inc = i / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    cos_i = cos(inc)
    t_ref = tau * per
    inv_per = 1.0 / per
    two_tdur = 2.0 / tdur
    z_lim_m = 1 + r_moon + 1e-6
    yp_min = max(abs(b_bary) - a_planet, 0.0)
    rp_lim = 1 + r_planet + 1e-6
    has_p = rp_lim > yp_min
    x_lim_p = a_planet + sqrt(max(rp_lim * rp_lim - yp_min * yp_min, 0.0)) + 1e-6
    A_x = a_moon * sqrt(cos_O * cos_O + sin_O * sin_O * cos_i * cos_i) * (1 + 1e-9)
    A_y = a_moon * sqrt(sin_O * sin_O + cos_O * cos_O * cos_i * cos_i) * (1 + 1e-9)
    has_m = abs(b_bary) - A_y < z_lim_m
    x_lim_m = z_lim_m + A_x + 1e-6
    half = tdur / 2
    los2 = np.empty(cap, dtype=np.int64)
    his2 = np.empty(cap, dtype=np.int64)
    c_t2 = np.empty(cap)
    c_d2 = np.empty(cap)
    n_out = 0
    for k in range(n_runs):
        lo = los[k]
        hi = his[k]
        cnt = hi - lo
        if cnt == 0:
            continue
        ct = c_t[k]
        cd = c_d[k]
        tr = time[lo:hi]
        h = _cell_samples(time, lo, hi, cell_dt)
        if h == 0:  # cells too short to pay: whole run
            los2[n_out] = lo
            his2[n_out] = hi
            c_t2[n_out] = ct
            c_d2[n_out] = cd
            n_out += 1
            continue
        sh = t0_shift_planet + cd
        # start guess of the searches (h > 0: cnt >= SUBRUN_H_MIN)
        t0r = tr[0]
        span = tr[cnt - 1] - t0r
        scale = (cnt - 1) / span if span > 0 else 0.0
        # planet interval [pl, ph) of the run
        pl = cnt
        ph = cnt
        if has_p:
            t_a = ct + (-x_lim_p + sh) * half
            t_b = ct + (x_lim_p + sh) * half
            pad = 1e-6 * tdur + 1e-12 * (abs(t_a) + abs(t_b))
            pl = _search_sorted(tr, t_a - pad, False, t0r, scale)
            ph = _search_sorted(tr, t_b + pad, True, t0r, scale)
        # moon range [ml, mh) of the run (at least 2 samples, or empty)
        ml = 0
        mh = 0
        if has_m:
            t_a = ct + (-x_lim_m + sh) * half
            t_b = ct + (x_lim_m + sh) * half
            pad = 1e-6 * tdur + 1e-12 * (abs(t_a) + abs(t_b))
            ml = _search_sorted(tr, t_a - pad, False, t0r, scale)
            mh = _search_sorted(tr, t_b + pad, True, t0r, scale)
            if mh - ml == 1:
                if mh < cnt:
                    mh += 1
                else:
                    ml -= 1
        start = n_out
        cur_s = -1
        cur_e = -1
        s = ml
        if pl < ph:
            e = min(pl, mh - 1)
            if e > ml:
                cur_s, cur_e, n_out = _moon_cells(
                    tr, ml, e, h, lo, t_ref, inv_per, cos_O, sin_O, cos_i, a_moon, ct,
                    two_tdur, sh, b_bary, z_lim_m, curv, cur_s, cur_e, los2, his2, n_out
                )
            cur_s, cur_e, n_out = _emit_interval(lo + pl, lo + ph, cur_s, cur_e, los2, his2, n_out)
            s = max(ph - 1, ml)
        if mh - 1 > s:
            cur_s, cur_e, n_out = _moon_cells(
                tr, s, mh - 1, h, lo, t_ref, inv_per, cos_O, sin_O, cos_i, a_moon, ct,
                two_tdur, sh, b_bary, z_lim_m, curv, cur_s, cur_e, los2, his2, n_out
            )
        if cur_s >= 0:
            los2[n_out] = cur_s
            his2[n_out] = cur_e
            n_out += 1
        for r in range(start, n_out):
            c_t2[r] = ct
            c_d2[r] = cd
    return los2[:n_out], his2[:n_out], c_t2[:n_out], c_d2[:n_out]


# P7b: eccentric moon orbit. Positions are linear in cos E, sin E
# (r cos nu = cos E - e, r sin nu = sqrt(1 - e^2) sin E; NbodyGradient /
# gefera form), so no true anomaly, tan or arctan is needed. The mean
# anomaly is reduced in turns (as P7): f = u - round(u) in [-1/2, 1/2),
# M = 2 pi |f| in [0, pi], and E(-M) = -E(M) replaces bug #2's reflection
# M > pi -> 2 pi - M (sin E changes sign). For e <= ECC_FAST_MAX Kepler's
# equation is solved branch-free, so the loop vectorises: Danby's start
# E = M + 0.85 e, four Halley steps d = 2 g g' / (2 g'^2 - g g'') (one
# division), sin / cos from trig.sincos_turns() for the first two and by
# rotation with trig.sincos_small(d) after that (|d| <= 0.27 there) and for
# the final E. Max |delta (cos E, sin E)| <= 2.4e-15 vs a long-double
# solution for 0 <= e <= 0.95, M in [1e-16, pi] (scratch_p7b/solver.py,
# proto2.py); ~12 instead of ~115 ns per sample. Above ECC_FAST_MAX the
# previous Markley (1995) starter plus a fourth-order correction (libm,
# scalar) is used.
ECC_FAST_MAX = 0.95
INV_TWO_PI = 1 / TWO_PI


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy", inline="always")
def _kepler_cs_fast(M, e):
    """(cos E, sin E) of the eccentric anomaly for 0 <= M <= pi and
    0 <= e <= ECC_FAST_MAX, branch-free (P7b)"""
    E = M + 0.85 * e
    s, c = sincos_turns(E * INV_TWO_PI)
    d = 0.0
    for it in range(4):
        if it == 1:
            s, c = sincos_turns(E * INV_TWO_PI)
        elif it > 1:
            ds, dc = sincos_small(d)
            s, c = s * dc - c * ds, c * dc + s * ds
        g = E - e * s - M
        fp = 1 - e * c
        d = 2 * g * fp / (2 * fp * fp - g * e * s)
        E = E - d
    ds, dc = sincos_small(d)
    return c * dc + s * ds, s * dc - c * ds


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _kepler_cs_markley(M, e):
    """(cos E, sin E) for 0 <= M <= pi, any 0 <= e < 1: Markley's starter
    and a fourth-order correction (the solver before P7b, libm per call)"""
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
    return cos(k), sin(k)


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def ellipse_ecc_z(a, per, e, tau, Omega, w, i, time, mass_ratio, x_bary, b_bary):
    """Eccentric moon orbit: (xm, ym, xp, yp, zm2, zp2), z2 = x^2 + y^2
    (P26, as ellipse_z()). P7b: positions linear in cos E, sin E, M reduced
    in turns, vectorised Kepler solver for e <= ECC_FAST_MAX (see above)."""
    cos_i = cos(i / 180 * pi)
    O = Omega / 180 * pi
    w_rad = w / 180 * pi
    cos_O = cos(O)
    sin_O = sin(O)
    cos_w = cos(w_rad)
    sin_w = sin(w_rad)
    # sky-plane direction = (ax X + bx Y, ay X + by Y), X = cos E - e,
    # Y = sqrt(1 - e^2) sin E (rotation by w in the orbit plane, then the
    # projection by Omega, i)
    ax = cos_O * cos_w - sin_O * cos_i * sin_w
    bx = -cos_O * sin_w - sin_O * cos_i * cos_w
    ay = sin_O * cos_w + cos_O * cos_i * sin_w
    by = -sin_O * sin_w + cos_O * cos_i * cos_w
    sq = sqrt((1 - e) * (1 + e))
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    fast = e <= ECC_FAST_MAX
    n = len(time)
    xm = np.empty(n)
    ym = np.empty(n)
    xp = np.empty(n)
    yp = np.empty(n)
    zm2 = np.empty(n)
    zp2 = np.empty(n)
    # cos E, sin E first, kept in zm2 / zp2 (two solver loops: a branch
    # between the solvers inside one loop made LLVM evaluate both per lane),
    # then the positions
    t_ref = tau * per
    inv_per = 1 / per
    if fast:
        for j in range(n):
            u = (time[j] - t_ref) * inv_per
            f = u - np.floor(u + 0.5)
            c, s = _kepler_cs_fast(TWO_PI * abs(f), e)
            zm2[j] = c
            zp2[j] = s if f >= 0 else -s
    else:
        for j in range(n):
            u = (time[j] - t_ref) * inv_per
            f = u - np.floor(u + 0.5)
            c, s = _kepler_cs_markley(TWO_PI * abs(f), e)
            zm2[j] = c
            zp2[j] = s if f >= 0 else -s
    for j in range(n):
        X = zm2[j] - e
        Y = sq * zp2[j]
        vector_x = ax * X + bx * Y
        vector_y = ay * X + by * Y
        xmj = vector_x * a_moon + x_bary[j]
        ymj = vector_y * a_moon + b_bary
        xpj = -vector_x * a_planet + x_bary[j]
        ypj = -vector_y * a_planet + b_bary
        xm[j] = xmj
        ym[j] = ymj
        xp[j] = xpj
        yp[j] = ypj
        zm2[j] = xmj * xmj + ymj * ymj
        zp2[j] = xpj * xpj + ypj * ypj
    return xm, ym, xp, yp, zm2, zp2
