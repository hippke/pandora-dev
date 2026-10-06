import numpy as np

from pandoramoon.pandora import pandora_loglike


class Likelihood(object):
    """Gaussian log-likelihood of a light curve for samplers (e.g. UltraNest).

    Precomputes the data terms once; each call evaluates the Pandora model
    only near transit (no full-length arrays are allocated per call):

        like = Likelihood(time, data, yerr, epoch_distance=..., ...)
        logL = like(u1, u2, R_star, per_bary, ..., M_moon)   # same order as pandora()

    time is the (supersampled) time grid as for pandora(); data and yerr are
    at output cadence (len(time) / supersampling_factor, rounded up). The
    result equals -0.5 * nansum(((flux_total - data) / yerr) ** 2): data or
    yerr that are NaN/inf get weight 0. All arithmetic is float64.

    For plots and videos use pandora_flux() (fluxes) or pandora()
    (fluxes and coordinates).
    """

    def __init__(
        self,
        time,
        data,
        yerr,
        epoch_distance,
        supersampling_factor=1,
        occult_small_threshold=0.01,
        hill_sphere_threshold=1.1,
        numerical_grid=25,
        cache=None,
        eclipse_method="analytic",
    ):
        if eclipse_method not in ("analytic", "pixelmap"):
            raise ValueError(
                'eclipse_method must be "analytic" or "pixelmap", got %r' % (eclipse_method,)
            )
        supersampling_factor = int(supersampling_factor)
        if supersampling_factor < 1:
            raise ValueError("supersampling_factor must be a positive integer")
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        n = len(self.time)
        n_out = n if supersampling_factor == 1 else -(-n // supersampling_factor)
        data = np.asarray(data, dtype=np.float64)
        yerr = np.broadcast_to(np.asarray(yerr, dtype=np.float64), data.shape)
        if data.shape != (n_out,):
            raise ValueError(
                "data has %d values, expected %d (time %d / supersampling_factor %d)"
                % (data.size, n_out, n, supersampling_factor)
            )
        valid = np.isfinite(data) & np.isfinite(yerr)
        if np.any(yerr[valid] <= 0):
            raise ValueError("yerr must be positive")
        self.data = np.ascontiguousarray(np.where(valid, data, 1.0))
        self.inv_err = np.ascontiguousarray(np.where(valid, 1.0 / np.where(valid, yerr, 1.0), 0.0))
        self.flat_terms = ((1.0 - self.data) * self.inv_err) ** 2
        self.chi2_flat = float(np.sum(self.flat_terms))
        self.epoch_distance = float(epoch_distance)
        self.supersampling_factor = supersampling_factor
        self.occult_small_threshold = float(occult_small_threshold)
        self.hill_sphere_threshold = float(hill_sphere_threshold)
        self.numerical_grid = int(numerical_grid)
        self.cache = cache
        self.eclipse_pixelmap = eclipse_method == "pixelmap"

    def __call__(
        self, u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
        i_moon, ecc_moon, w_moon, M_moon
    ):
        return pandora_loglike(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
            Omega_moon, i_moon, ecc_moon, w_moon, M_moon, self.epoch_distance,
            self.supersampling_factor, self.occult_small_threshold,
            self.hill_sphere_threshold, self.numerical_grid, self.time, self.data,
            self.inv_err, self.flat_terms, self.chi2_flat, self.cache,
            self.eclipse_pixelmap,
        )

    loglike = __call__
