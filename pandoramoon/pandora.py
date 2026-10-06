import numpy as np
from numpy import sqrt, pi, arcsin, cos, degrees
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os import path

# Pandora
from pandoramoon.eclipse import eclipse
from pandoramoon.ellipse import ellipse, ellipse_ecc
from pandoramoon.occult import occult, occult_small, occult_hybrid, create_occult_cache, read_occult_cache
from pandoramoon.helpers import resample
from pandoramoon.grids import timegrid, x_bary_grid, x_bary_window, x_bary_window_sorted


class model_params(object):
    def __init__(self):

        # Star parameters
        self.u1 = None
        self.u2 = None
        self.R_star = None

        # Planet parameters
        self.per_bary = None
        self.a_bary = None
        self.r_planet = None
        self.b_bary = None
        self.w_bary = 0
        self.ecc_bary = 0
        self.t0_bary = None
        self.t0_bary_offset = None
        self.M_planet = None

        # Moon parameters
        self.r_moon = None
        self.per_moon = None
        self.tau_moon = None
        self.Omega_moon = None
        self.i_moon = None
        self.ecc_moon = 0
        self.w_moon = 0
        self.M_moon = None

        # Other model parameters
        self.epochs = None
        self.epoch_duration = None
        self.cadences_per_day = None
        self.epoch_distance = None
        self.supersampling_factor = 1
        self.occult_small_threshold = 0.01
        self.hill_sphere_threshold = 1.1
        self.numerical_grid = 25
        # Mutual eclipses with a body on the stellar limb:
        # "analytic" (exact 3-circle overlap, default) or "pixelmap"
        # (legacy pixel raster of size numerical_grid)
        self.eclipse_method = "analytic"
        self.time = None


class time(object):
    def __init__(self, params):
        self.t0_bary = params.t0_bary
        self.epochs = params.epochs
        self.epoch_duration = params.epoch_duration
        self.cadences_per_day = params.cadences_per_day
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor

    def grid(self):
        return timegrid(
            self.t0_bary, 
            self.epochs, 
            self.epoch_duration, 
            self.cadences_per_day, 
            self.epoch_distance, 
            self.supersampling_factor
        )


class moon_model(object):
    def __init__(self, params):

        # Star parameters
        self.u1 = params.u1
        self.u2 = params.u2
        self.R_star = params.R_star

        # Planet parameters
        self.per_bary = params.per_bary
        self.a_bary = params.a_bary
        self.r_planet = params.r_planet
        self.b_bary = params.b_bary
        self.w_bary = params.w_bary
        self.ecc_bary = params.ecc_bary
        self.t0_bary = params.t0_bary
        self.t0_bary_offset = params.t0_bary_offset
        self.M_planet = params.M_planet

        # Moon parameters
        self.r_moon = params.r_moon
        self.per_moon = params.per_moon
        self.tau_moon = params.tau_moon
        self.Omega_moon = params.Omega_moon
        self.i_moon = params.i_moon
        self.ecc_moon = params.ecc_moon
        self.w_moon = params.w_moon
        self.M_moon = params.M_moon

        # Other model parameters
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor
        self.occult_small_threshold = params.occult_small_threshold
        self.hill_sphere_threshold = params.hill_sphere_threshold
        self.numerical_grid = params.numerical_grid
        self.eclipse_method = getattr(params, "eclipse_method", "analytic")
        self._eclipse_pixelmap()  # validate early
        self.time = params.time
        self.cache = create_occult_cache(self.u1, self.u2, dim=300)

    def _eclipse_pixelmap(self):
        if self.eclipse_method == "analytic":
            return False
        if self.eclipse_method == "pixelmap":
            return True
        raise ValueError(
            'eclipse_method must be "analytic" or "pixelmap", got %r' % (self.eclipse_method,)
        )

    def video(
        self,
        time,
        limb_darkening=True,
        teff=6000,
        planet_color="black",
        moon_color="black",
        ld_circles=100,
    ):
        (
            self.flux_planet,
            self.flux_moon,
            self.flux_total,
            self.px,
            self.py,
            self.mx,
            self.my
        ) = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time,
            eclipse_pixelmap=self._eclipse_pixelmap()
        )
        # Build video with matplotlib
        fig = plt.figure(figsize=(5, 5))
        axes = fig.add_subplot(111)
        plt.axis("off")
        plt.style.use("dark_background")
        plt.gcf().gca().add_artist(plt.Circle((0, 0), 5, color="black"))
        if limb_darkening:
            if teff > 12000:
                teff = 12000
            if teff < 2300:
                teff = 2300
            data_dir = path.join(path.join(path.dirname(__file__)), "star_colors.csv")
            star_colors = np.genfromtxt(data_dir, delimiter=",")
            row = np.argmax(star_colors[:, 0] >= teff)
            r_star = star_colors[row, 1]
            g_star = star_colors[row, 2]
            b_star = star_colors[row, 3]
            for i in reversed(range(ld_circles)):
                impact = (i + 1) / ld_circles
                r_mid = (i + 0.5) / ld_circles
                m = sqrt(1 - min(r_mid ** 2, 1))
                ld = 1 - self.u1 * (1 - m) - self.u2 * (1 - m) ** 2
                r = r_star * ld
                g = g_star * ld
                b = b_star * ld
                sun = plt.Circle((0, 0), impact, color=(r, g, b))
                plt.gcf().gca().add_artist(sun)
        else:
            plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color="yellow"))

        axes.set_xlim(-1.05, 1.05)
        axes.set_ylim(-1.05, 1.05)
        (moon,) = axes.plot(
            self.mx[0],
            self.my[0],
            "o",
            color=planet_color,
            markerfacecolor=moon_color,
            markeredgecolor=moon_color,
            markersize=260 * self.r_moon,
        )
        (planet,) = axes.plot(
            self.px[0],
            self.py[0],
            "o",
            color=planet_color,
            markeredgecolor=planet_color,
            markerfacecolor=planet_color,
            markersize=260 * self.r_planet,
        )

        def ani(coords):
            mx, my, px, py = coords
            moon.set_data([mx], [my])
            planet.set_data([px], [py])
            pbar.update(1)
            return moon, planet

        def frames():
            for mx, my, px, py in zip(self.mx, self.my, self.px, self.py):
                yield mx, my, px, py

        pbar = tqdm(total=len(self.mx))
        ani = FuncAnimation(fig, ani, frames=frames, save_count=int(1e15), blit=True)
        return ani


    def light_curve(self, time):
        flux_planet, flux_moon, flux_total, px, py, mx, my = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time,
            self.cache,
            eclipse_pixelmap=self._eclipse_pixelmap()
        )
        return flux_total, flux_planet, flux_moon

    def coordinates(self, time):
        flux_planet, flux_moon, flux_total, px, py, mx, my = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time,
            eclipse_pixelmap=self._eclipse_pixelmap()
        )
        return px, py, mx, my


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    numerical_grid,
    time,
    cache=None,
    eclipse_pixelmap=False
):
    # eclipse_pixelmap=False: limb mutual eclipses via the exact analytic
    # three-circle overlap (default); True: legacy pixelart() raster.

    # Make sure to work with floats. Large values as ints would overflow.
    R_star = float(R_star)
    per_bary = float(per_bary)
    a_bary = float(a_bary)
    r_planet = float(r_planet)
    b_bary = float(b_bary)
    t0_bary = float(t0_bary)
    t0_bary_offset = float(t0_bary_offset)
    M_planet = float(M_planet)
    r_moon = float(r_moon)
    per_moon = float(per_moon)
    tau_moon = float(tau_moon)
    Omega_moon = float(Omega_moon)
    i_moon = float(i_moon)
    M_moon = float(M_moon)

    a_moon, unphysical = _moon_setup(
        R_star, per_bary, a_bary, M_planet, M_moon, per_moon, ecc_moon,
        r_planet, r_moon, hill_sphere_threshold
    )
    x_bary = x_bary_grid(
        time, 
        a_bary, 
        per_bary, 
        t0_bary, 
        t0_bary_offset, 
        epoch_distance, 
        ecc_bary, 
        w_bary
    )
    xm, ym, xp, yp = _orbit(
        time, x_bary, unphysical, a_moon, per_moon, tau_moon, Omega_moon,
        w_moon, i_moon, ecc_moon, M_moon / M_planet, b_bary
    )

    # Occultations and mutual eclipses only for samples that can be in
    # transit; all others have flux exactly 1 (see _cull_window)
    idx = _cull_window(
        x_bary, _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache)
    )
    if len(idx) == len(time):
        flux_planet, flux_moon = _window_fluxes(
            xp, yp, xm, ym, unphysical, r_planet, r_moon, u1, u2,
            occult_small_threshold, numerical_grid, cache, eclipse_pixelmap
        )
    else:
        flux_planet_w, flux_moon_w = _window_fluxes(
            xp[idx], yp[idx], xm[idx], ym[idx], unphysical, r_planet, r_moon,
            u1, u2, occult_small_threshold, numerical_grid, cache, eclipse_pixelmap
        )
        flux_planet, flux_moon = _scatter(len(time), idx, flux_planet_w, flux_moon_w)
    flux_planet, flux_moon, flux_total = _total_and_resample(
        flux_planet, flux_moon, supersampling_factor
    )
    return flux_planet, flux_moon, flux_total, xp, yp, xm, ym


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_flux(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    numerical_grid,
    time,
    cache=None,
    eclipse_pixelmap=False
):
    """Same arguments and fluxes as pandora(), but returns only
    (flux_planet, flux_moon, flux_total) and computes the orbit only for
    samples that can be in transit. float64, full length (output cadence).
    For likelihoods, Likelihood (no full-length arrays) or pandora_deficit()
    (float32) are faster."""
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, numerical_grid, time,
        cache, eclipse_pixelmap, False
    )
    out_idx, fp, fm, ft = _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
    n_out = _n_out(len(time), supersampling_factor)
    return (
        _densify(n_out, out_idx, fp, 1.0),
        _densify(n_out, out_idx, fm, 1.0),
        _densify(n_out, out_idx, ft, 1.0),
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_deficit(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    numerical_grid,
    time,
    cache=None,
    eclipse_pixelmap=False
):
    """Same arguments as pandora_flux(), but returns the flux deficits
    (1 - flux_planet, 1 - flux_moon, 1 - flux_total) as float32 arrays
    (full length, output cadence; exactly 0 out of transit).

    Half the memory of float64 flux arrays. The deficit, unlike the flux,
    keeps float32's full relative precision (2^-24 ~ 6e-8 of the deficit
    itself, i.e. ~1e-9 absolute for a 1 % transit); float32 flux would be
    quantised to 6e-8 absolute near 1. All computations are float64; only
    the stored result is rounded. In a chi^2 use (1 - data) - deficit."""
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, numerical_grid, time,
        cache, eclipse_pixelmap, False
    )
    out_idx, fp, fm, ft = _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
    n_out = _n_out(len(time), supersampling_factor)
    return (
        _densify_deficit32(n_out, out_idx, fp),
        _densify_deficit32(n_out, out_idx, fm),
        _densify_deficit32(n_out, out_idx, ft),
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_window(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    numerical_grid,
    time,
    cache=None,
    eclipse_pixelmap=False,
    time_sorted=False
):
    """Sparse model output: same arguments as pandora_flux(), returns
    (idx, flux_planet, flux_moon, flux_total), where idx (int64, ascending)
    are the output cadences near transit and the float64 fluxes are given
    there only. All other output cadences have flux exactly 1 (no full-length
    array is created). pandora_flux() is this, densified.
    time_sorted=True: time is ascending (not checked); the window is then
    found by binary search per epoch (same result, no pass over all
    samples). Wrong results if time is not sorted."""
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, numerical_grid, time,
        cache, eclipse_pixelmap, time_sorted
    )
    return _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_loglike(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, numerical_grid, time,
    data, inv_err, flat_terms, chi2_flat, cache, eclipse_pixelmap, time_sorted
):
    """Gaussian log-likelihood -chi^2/2 of the total flux, without
    full-length arrays: chi^2 = chi2_flat + sum over window cadences of
    [((f - data) inv_err)^2 - flat_terms], where flat_terms =
    ((1 - data) inv_err)^2 and chi2_flat = sum(flat_terms) (model = 1
    everywhere) are precomputed per dataset. Use it via Likelihood."""
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, numerical_grid, time,
        cache, eclipse_pixelmap, time_sorted
    )
    out_idx, ft = _window_cadences_total(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
    chi2 = chi2_flat
    for k in range(len(out_idx)):
        i = out_idx[k]
        r = (ft[k] - data[i]) * inv_err[i]
        chi2 += r * r - flat_terms[i]
    return -0.5 * chi2


# ----------------------------------------------------------------------------
# Building blocks of pandora()
# ----------------------------------------------------------------------------
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _moon_setup(
    R_star, per_bary, a_bary, M_planet, M_moon, per_moon, ecc_moon, r_planet,
    r_moon, hill_sphere_threshold
):
    """Planet-moon semi-major axis [R_star] and the plausibility flag
    (unphysical moon orbits are moved out of transit)"""
    # Calculate moon period around planet
    G = 6.67408e-11
    day = 60 * 60 * 24
    a_moon = (
        G * (M_planet + M_moon) / (2 * pi / (per_moon * day)) ** 2
    ) ** (1 / 3)
    a_moon /= R_star

    # Check physical plausibility of a_moon
    # Should be inside [Roche lobe, Hill sphere] plus/minus some user-set margin
    M_star = ((4 * pi ** 2 / G) * ((a_bary * R_star) ** 3)) / (per_bary * day) ** 2
    r_hill = a_bary * (M_planet / (3 * M_star)) ** (1 / 3)
    r_hill_fraction = a_moon * (1 + ecc_moon) / r_hill
    if r_hill_fraction > hill_sphere_threshold:
        unphysical = True
    else:
        unphysical = False

    # Roche lobe: Not used here, because we don't know densities of planet and moon
    # Instead of taking density guesses, we just demand a_moon > (R_planet + R_moon)
    if a_moon * (1 - ecc_moon) < (r_planet + r_moon):
        unphysical = True
    return a_moon, unphysical


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _orbit(
    time, x_bary, unphysical, a_moon, per_moon, tau_moon, Omega_moon, w_moon,
    i_moon, ecc_moon, mass_ratio, b_bary
):
    """Sky-plane coordinates xm, ym, xp, yp [R_star] of moon and planet"""
    # Unphysical moon orbit: Keep planet, but put moon at far out of transit position
    if unphysical:
        bignum = 1e8
        xp = x_bary
        yp = np.full(len(x_bary), b_bary)
        xm = np.full(len(x_bary), bignum)
        ym = xm.copy()
    else:  # valid, physical system
        if ecc_moon == 0:
            xm, ym, xp, yp = ellipse(
                a=a_moon,
                per=per_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                i=i_moon,
                time=time,
                x_bary=x_bary,
                mass_ratio=mass_ratio,
                b_bary=b_bary,
            )
        else:
            xm, ym, xp, yp = ellipse_ecc(
                a=a_moon,
                per=per_moon,
                e=ecc_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                w=w_moon,
                i=i_moon,
                time=time,
                mass_ratio=mass_ratio,
                x_bary=x_bary,
                b_bary=b_bary,
            )
    return xm, ym, xp, yp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache):
    """Conservative bound on |x_bary| [R_star] beyond which neither body can
    touch the star, so that all fluxes are exactly 1.

    Both bodies are offset from the barycentre by at most a_moon * (1 + e)
    on the sky (|projected unit vector| <= 1, orbital radius <= a (1 + e);
    the planet's share is a q/(1+q), the moon's a/(1+q)). A body has
    flux exactly 1 for z >= 1 + r (occult_small/occult_hybrid) or, with the
    occultation cache, for z >= 1.1 (bilinear interpolation reaches past
    1 + r up to the grid end). Mutual eclipses only modify a moon flux < 1.
    The margin 1e-6 covers float round-off."""
    r_limit = 1 + r_planet
    if not unphysical:  # moon at 1e8 otherwise
        r_limit = 1 + max(r_planet, r_moon)
    if cache is not None:
        r_limit = max(r_limit, 1.1)
    if unphysical:  # planet sits at the barycentre
        return r_limit + 1e-6
    return r_limit + a_moon * (1 + abs(ecc_moon)) + 1e-6


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _cull_window(x_bary, reach):
    """Indices of samples with |x_bary| < reach (NaN is kept: not culled)"""
    n = len(x_bary)
    idx = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        if not (abs(x_bary[i]) >= reach):
            idx[m] = i
            m += 1
    return idx[:m]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_fluxes(
    xp, yp, xm, ym, unphysical, r_planet, r_moon, u1, u2,
    occult_small_threshold, numerical_grid, cache, eclipse_pixelmap
):
    """Planet and moon flux (incl. mutual eclipses) for the given samples"""
    # Distances of planet and moon from (0,0) = center of star
    # Not sufficient to only calculate z in ellipse func: 
    # We also need full coordinates to determine mutual eclipses below
    z_planet = sqrt(xp ** 2 + yp ** 2)
    if unphysical:
        z_moon = xm.copy()
    else:
        z_moon = sqrt(xm ** 2 + ym ** 2)

    # Cached Mandel-Agol occultation model for planet < 0.1, else hybrid
    if cache is not None and r_planet < 0.1:
        if r_planet < cache[1][0]:
            # k below the cache grid: use exact small-body approximation
            flux_planet = occult_small(zs=z_planet, k=r_planet, u1=u1, u2=u2)
        else:
            flux_planet = read_occult_cache(
                zs_target=z_planet, 
                k=r_planet, 
                cache=cache
            )
    else:
        flux_planet = occult_hybrid(zs=z_planet, u1=u1, u2=u2, k=r_planet)
        #flux_planet = occult(zs=z_planet, u1=u1, u2=u2, k=r_planet)
    
    # For moon transit: User can "set occult_small_threshold > 0"
    if r_moon < occult_small_threshold:
        flux_moon = occult_small(zs=z_moon, k=r_moon, u1=u1, u2=u2)
    elif cache is not None and r_moon < 0.1:
        if r_moon < cache[1][0]:
            # k below the cache grid: use exact small-body approximation
            flux_moon = occult_small(zs=z_moon, k=r_moon, u1=u1, u2=u2)
        else:
            flux_moon = read_occult_cache(zs_target=z_moon, k=r_moon, cache=cache)
    else:
        flux_moon = occult_hybrid(zs=z_moon, k=r_moon, u1=u1, u2=u2)

    # Mutual planet-moon occultations
    if not unphysical:# and eclipses_occur:
        flux_moon = eclipse(
            xp, yp, xm, ym, r_planet, r_moon, flux_moon, numerical_grid, eclipse_pixelmap
        )
    return flux_planet, flux_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _scatter(n, idx, flux_planet_w, flux_moon_w):
    """Full-length flux arrays: 1 everywhere except at the window indices"""
    flux_planet = np.ones(n)
    flux_moon = np.ones(n)
    for j in range(len(idx)):
        flux_planet[idx[j]] = flux_planet_w[j]
        flux_moon[idx[j]] = flux_moon_w[j]
    return flux_planet, flux_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _total_and_resample(flux_planet, flux_moon, supersampling_factor):
    flux_total = flux_moon + flux_planet - 1

    # Supersampling downconversion
    if supersampling_factor > 1:
        flux_planet = resample(flux_planet, supersampling_factor)
        flux_moon = resample(flux_moon, supersampling_factor)
        flux_total = resample(flux_total, supersampling_factor)
    return flux_planet, flux_moon, flux_total


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_model(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, occult_small_threshold,
    hill_sphere_threshold, numerical_grid, time, cache, eclipse_pixelmap,
    time_sorted
):
    """Culled model core: (ascending sample indices of the window, planet
    flux, moon flux there). Samples outside the window have flux exactly 1.
    time_sorted=True (time ascending, caller's guarantee): the window is
    found by binary search per epoch instead of a pass over all samples."""
    R_star = float(R_star)
    per_bary = float(per_bary)
    a_bary = float(a_bary)
    r_planet = float(r_planet)
    b_bary = float(b_bary)
    t0_bary = float(t0_bary)
    t0_bary_offset = float(t0_bary_offset)
    M_planet = float(M_planet)
    r_moon = float(r_moon)
    per_moon = float(per_moon)
    tau_moon = float(tau_moon)
    Omega_moon = float(Omega_moon)
    i_moon = float(i_moon)
    M_moon = float(M_moon)

    a_moon, unphysical = _moon_setup(
        R_star, per_bary, a_bary, M_planet, M_moon, per_moon, ecc_moon,
        r_planet, r_moon, hill_sphere_threshold
    )
    # Barycentre positions only for samples that can be in transit
    reach = _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache)
    if time_sorted:
        idx, x_bary = x_bary_window_sorted(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    else:
        idx, x_bary = x_bary_window(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    xm, ym, xp, yp = _orbit(
        time[idx], x_bary, unphysical, a_moon, per_moon, tau_moon,
        Omega_moon, w_moon, i_moon, ecc_moon, M_moon / M_planet, b_bary
    )
    flux_planet_w, flux_moon_w = _window_fluxes(
        xp, yp, xm, ym, unphysical, r_planet, r_moon, u1, u2,
        occult_small_threshold, numerical_grid, cache, eclipse_pixelmap
    )
    return idx, flux_planet_w, flux_moon_w


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _n_out(n, supersampling_factor):
    """Number of output cadences for n samples (as resample())"""
    if supersampling_factor <= 1:
        return n
    return int(np.ceil(n / supersampling_factor))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_cadences(n, idx, flux_planet_w, flux_moon_w, supersampling_factor):
    """Output cadences touched by the (ascending) window sample indices, and
    planet/moon/total flux there; all other output cadences have flux 1.
    With supersampling, a touched cadence is the mean over its block with
    flux 1 at non-window samples (as resample() of the full array).
    Same values as _scatter() followed by _total_and_resample() up to
    rounding (~1e-16), without full-length temporaries: the block mean is
    accumulated as a deficit, 1 - sum(1 - f)/length (non-window samples
    contribute 0, so a block of exact 1s stays exactly 1), and the total is
    1 - (deficit_planet + deficit_moon)."""
    m = len(idx)
    if supersampling_factor <= 1:
        flux_total_w = np.empty(m)
        for j in range(m):
            flux_total_w[j] = flux_moon_w[j] + flux_planet_w[j] - 1
        return idx, flux_planet_w, flux_moon_w, flux_total_w

    factor = supersampling_factor
    out_idx = np.empty(m, dtype=np.int64)
    flux_planet = np.empty(m)
    flux_moon = np.empty(m)
    flux_total = np.empty(m)
    j = 0
    k = 0
    while j < m:
        out = idx[j] // factor
        start = out * factor
        end = start + factor
        if end > n:
            end = n
        length = end - start
        dp = 0.0
        dm = 0.0
        while j < m and idx[j] < end:
            dp += 1.0 - flux_planet_w[j]
            dm += 1.0 - flux_moon_w[j]
            j += 1
        dp /= length
        dm /= length
        out_idx[k] = out
        flux_planet[k] = 1.0 - dp
        flux_moon[k] = 1.0 - dm
        flux_total[k] = 1.0 - (dp + dm)
        k += 1
    return out_idx[:k], flux_planet[:k], flux_moon[:k], flux_total[:k]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_cadences_total(n, idx, flux_planet_w, flux_moon_w, supersampling_factor):
    """Total flux only, as (out_idx, flux_total) of _window_cadences()
    (same values; for the likelihood, which needs no planet/moon outputs)."""
    m = len(idx)
    if supersampling_factor <= 1:
        flux_total_w = np.empty(m)
        for j in range(m):
            flux_total_w[j] = flux_moon_w[j] + flux_planet_w[j] - 1
        return idx, flux_total_w

    factor = supersampling_factor
    out_idx = np.empty(m, dtype=np.int64)
    flux_total = np.empty(m)
    j = 0
    k = 0
    while j < m:
        out = idx[j] // factor
        start = out * factor
        end = start + factor
        if end > n:
            end = n
        length = end - start
        dp = 0.0
        dm = 0.0
        while j < m and idx[j] < end:
            dp += 1.0 - flux_planet_w[j]
            dm += 1.0 - flux_moon_w[j]
            j += 1
        dp /= length
        dm /= length
        out_idx[k] = out
        flux_total[k] = 1.0 - (dp + dm)
        k += 1
    return out_idx[:k], flux_total[:k]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _densify(n_out, out_idx, values, fill):
    """Full-length float64 array: fill, with values at out_idx"""
    out = np.full(n_out, fill)
    for k in range(len(out_idx)):
        out[out_idx[k]] = values[k]
    return out


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _densify_deficit32(n_out, out_idx, flux):
    """Full-length float32 deficit 1 - flux (0 elsewhere)"""
    out = np.zeros(n_out, dtype=np.float32)
    for k in range(len(out_idx)):
        out[out_idx[k]] = 1.0 - flux[k]
    return out
