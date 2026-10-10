Fitting: evidence and posterior
===============================

Pandora contains its own nested sampler, ``NestedSampler``. It fits a light curve with a planet+moon
model and with a planet-only model and returns both evidences and their ratio (the Bayes factor). It
also returns the posterior of every parameter and checks its own convergence. This page describes the
interface and the background needed to use it well.

.. contents::
   :local:
   :depth: 2


Quick start
-----------

::

   import numpy as np
   import pandoramoon as pandora
   from pandoramoon import NestedSampler, Uniform, LogUniform, Normal

   R_sun = 696_342_000  # [m]
   # time, flux, flux_err: your light curve (e.g. segments of +-1 d around each transit)
   like = pandora.Likelihood(time, flux, flux_err, epoch_distance=365.26, hill_sphere_threshold=1.0)

   priors = dict(
       q1=Uniform(0, 1), q2=Uniform(0, 1),                 # limb darkening (Kipping 2013)
       R_star=Uniform(0.4 * R_sun, 1.2 * R_sun),
       per_bary=Uniform(360.25, 370.25), a_bary=Uniform(210, 220), r_planet=Uniform(0.07, 0.11),
       b_bary=Uniform(0, 1), t0_bary=11.0, t0_bary_offset=Uniform(-0.05, 0.05),
       M_planet=Uniform(1.5e27, 2.5e27),
       r_moon=Uniform(0.01, 0.07), per_moon=Uniform(0, 50), tau_moon=Uniform(0, 1),
       Omega_moon=Uniform(0, 180), i_moon=Uniform(0, 180), M_moon=Uniform(1, 1e27),
   )
   sampler = NestedSampler(like, priors, log_dir="fit_out")
   results = sampler.run()
   sampler.print_results()
   sampler.plot_corner()

   print(results["logz"], results["logzerr"])        # planet+moon evidence ln Z
   print(results["planet"]["logz"])                   # planet-only evidence
   print(results["dlogz"], results["dlogzerr"])      # Bayes factor ln Z(moon) - ln Z(planet)
   samples = results["samples"]                      # equal-weight posterior, columns results["paramnames"]

With the notebook's data (3 transits, 30-min cadence, 500 ppm) the planet+moon fit takes 2–4 minutes on
16 cores and the planet-only fit a few seconds.


Interface
---------

.. class:: NestedSampler(likelihood, priors, model="moon", compare_planet=True, moon_phase="auto", ld_correlation=0.0, log_dir=None, resume="subfolder", threads=None, seed=1, verbose=True)

   :likelihood: A :class:`pandoramoon.Likelihood`. It holds the data, the errors and Pandora's model
      settings (``supersampling_factor``, ``hill_sphere_threshold``, ``occult_small_threshold``,
      ``epoch_distance``). The occultation ``cache`` is not supported (use ``cache=None``).
   :priors: *(dict)* For each model parameter, either a prior (:class:`Uniform`, :class:`LogUniform`,
      :class:`Normal`) or a number, which fixes the parameter. The names and units are those of
      Pandora's model (see :doc:`Interface`):
      ``R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0_bary, t0_bary_offset,
      M_planet, r_moon, per_moon, tau_moon, Omega_moon, i_moon, ecc_moon, w_moon, M_moon``, plus the
      limb darkening as either ``q1, q2`` (Kipping 2013) or ``u1, u2``.
      ``w_bary``, ``ecc_bary``, ``ecc_moon`` and ``w_moon`` default to ``0`` (circular orbits).
      For ``model="planet"`` the moon parameters are not needed. Neither are ``R_star`` and
      ``M_planet``, which act only through the moon's orbit.
   :model: *(str)* ``"moon"`` (planet + moon, default) or ``"planet"``.
   :compare_planet: *(bool)* For ``model="moon"``: also fit the planet-only model with the same priors
      for the shared parameters, and return the Bayes factor. Default ``True``.
   :moon_phase: ``"auto"`` (default), ``True`` or ``False``. Sample the moon's period and phase in the
      phase-at-transit coordinates (see `Moon orbit: phase-at-transit coordinates`_). With ``"auto"``
      they are used whenever the priors allow it. ``True`` raises an error if they cannot be used.
      ``False`` samples ``per_moon`` and ``tau_moon`` directly (not recommended for moons; see below).
   :ld_correlation: *(float)* Correlation coefficient between :class:`Normal` priors on ``u1`` and
      ``u2`` (e.g. limb darkening from stellar models). Default ``0``.
   :log_dir: *(str)* Output folder (see `Files`_). ``None``: no files.
   :resume: ``"subfolder"`` (default, as in UltraNest: a new folder ``run1``, ``run2``, … inside
      *log_dir* for every new sampler), ``"resume"`` (continue or reuse the fit stored in *log_dir*)
      or ``"overwrite"``.
   :threads: *(int)* Number of runs in parallel. Default: all cores available to the process.
   :seed: *(int)* Random seed. A fit is reproducible for a given seed and number of threads.
   :verbose: *(bool)* Print the progress of the checks.

   .. method:: run(accuracy="high", **kwargs)

      Runs the fit (for ``model="moon"``: first the planet-only fit, then the planet+moon fit) and
      returns the results dictionary (also stored as ``sampler.results``).

      :accuracy: ``"high"`` (default): statistical error of ln Z ≤ 0.1, posterior reproducible to a
         KS distance ≤ 0.05, 1000 live points per run, at least 16 runs. ``"medium"``: ≤ 0.2 and
         ≤ 0.08, 500 live points per run, at least 8 runs; about 2–5× faster.
      :kwargs: Passed to :func:`pandoramoon.nss.fit`, e.g. ``nlive``, ``lz_tol``, ``ks_tol``,
         ``min_runs``, ``max_runs`` (default 96), ``nsteps`` (initial chain length, default 4 × the
         number of sampled coordinates).

   .. method:: print_results()

      Prints ln Z ± error, whether the checks passed, and median, 68 % interval, mean and standard
      deviation of every parameter (for ``model="moon"`` also the planet-only fit and the Bayes factor).

   .. method:: plot_corner(model=None, filename=None, **kwargs)

      Corner plot of the posterior (needs the ``corner`` package). With a *log_dir*, it is saved as
      ``plots/corner.pdf``. ``model="planet"`` plots the planet-only fit.


Priors
~~~~~~

.. class:: Uniform(lo, hi)

   Flat on [*lo*, *hi*].

.. class:: LogUniform(lo, hi)

   Flat in ln *x* on [*lo*, *hi*], 0 < *lo* < *hi*. Typical for scale parameters spanning orders of
   magnitude (e.g. moon mass).

.. class:: Normal(mean, sigma, lo=-inf, hi=inf)

   Gaussian, optionally truncated to [*lo*, *hi*] and renormalised. Typical for parameters measured
   elsewhere (stellar radius, limb darkening, planet period).

Every prior is applied exactly. The sampler draws in a unit cube, and the inverse cumulative
distribution function of each prior maps the cube to the parameter. The evidence is therefore that of
your prior.

**Limb darkening.**

- ``q1, q2``: Kipping's parametrisation. ``q1, q2 ~ Uniform(0, 1)`` is the same as a flat prior over all
  physical quadratic laws.
- ``u1, u2``: any of the priors above, e.g. ``u1=Normal(0.41, 0.05), u2=Normal(0.26, 0.05)``. The prior
  is truncated to Pandora's physical region (u1 ≥ 0, u1 + u2 ≤ 1, u1 + 2 u2 ≥ 0) and renormalised, so
  the evidence stays correct. ``ld_correlation`` correlates two untruncated Normal priors.

**Bayes factors depend on the priors.** The evidence is the likelihood averaged over the prior, so wider
priors lower it. Comparing planet+moon with planet-only is meaningful only with the same priors on the
shared parameters. ``NestedSampler`` guarantees this by building both models from one prior dictionary.
For example, on the notebook's data the Bayes factor is 131.6 with the planet box a_bary ∈ [210, 220],
r_planet ∈ [0.07, 0.11]. A planet-only fit with the notebook's wide planet priors (a_bary ∈ [1, 301],
r_planet ∈ [0.001, 0.201]) has an evidence 1.2 lower.


Results
~~~~~~~

The results dictionary uses UltraNest's keys:

:logz, logzerr: ln Z and its statistical error (the scatter of ln Z between the independent runs,
   divided by √(number of runs)).
:logzerr_bs: Bootstrap error over the runs (agrees with ``logzerr``).
:logz_single: Mean ln Z of the single runs.
:paramnames: Names of the free parameters (the columns of the samples).
:samples: Equal-weight posterior samples, shape (n, number of free parameters).
:weighted_samples: Dictionary of all points of the merged runs: ``points`` (parameters),
   ``upoints`` (sampled unit-cube coordinates), ``weights`` (posterior weights, sum 1), ``logl``
   (log-likelihood).
:posterior: Dictionary of lists per parameter: ``mean``, ``stdev``, ``median``, ``errlo``, ``errup``
   (15.9 % and 84.1 % quantiles).
:maxlikelihood: The point with the highest likelihood found: ``logl``, ``point``,
   ``point_untransformed``.
:ess: Effective sample size of the posterior.
:H: Information gain from the prior to the posterior [nats].
:ncall, niter: Likelihood calls (including the check runs) and nested-sampling iterations.
:insertion_order_test: Insertion-index test (Fowlie, Handley & Su 2020), pooled over runs.
:passed: ``True`` if all convergence checks passed (see `Self-checks`_). If ``False``, the fit
   stopped at ``max_runs``; the result is usable but its error is not guaranteed.
:nsteps, runs, nlive, n_escalations, history: Final chain length, number of merged runs, live points
   per run, number of chain-length doublings, and the history of the checks.

For ``model="moon"`` with the planet comparison, the dictionary also has:

:planet: The planet-only results (the same keys).
:dlogz, dlogzerr: The Bayes factor ln Z(moon) − ln Z(planet) and its error.
:logz_excluded_correction: The (tiny) exact correction for the short-period part of the moon-period
   prior (see `Moon orbit: phase-at-transit coordinates`_).

A rough guide to Bayes factors (Kass & Raftery 1995): ΔlnZ > 5 is strong evidence for a moon.
Calibration on injected white-noise light curves: light curves without a moon gave ΔlnZ between about −6
and +2. The moon model's Occam penalty makes negative values the norm.


Files
~~~~~

With *log_dir*, the layout follows UltraNest:

====================================  ======================================================================
``info/results.json``                 all results except the samples (plus the priors)
``info/post_summary.csv``             mean, stdev, median, errlo, errup of each parameter (one row)
``chains/equal_weighted_post.txt``    equal-weight posterior samples, header = parameter names
``chains/weighted_post.npz``          weighted samples (``points``, ``upoints``, ``weights``, ``logl``;
                                      npz instead of text, because merged fits have up to ~10⁶ points)
``plots/corner.pdf``                  from ``plot_corner()``
``runs/``                             every finished nested-sampling run (for resume; ~3 MB per run)
``info/fit_state.json``               current chain length and a hash of data, priors and settings
``planet/``                           the planet-only fit, same layout
====================================  ======================================================================

**Resume.** A new ``NestedSampler`` with ``resume="resume"`` and the same *log_dir* reloads the stored
runs, evaluates the checks, and continues with new runs if needed. A finished fit is returned at once.
If the data, priors or settings differ from the stored fit, an error is raised.


Background
----------

Nested sampling
~~~~~~~~~~~~~~~

The evidence Z = ∫ L(θ) π(θ) dθ is the likelihood averaged over the prior. Nested sampling (Skilling
2006) computes it, and the posterior along the way, as follows:

- It keeps *n* "live" points drawn from the prior.
- It repeatedly removes the point with the lowest likelihood L* and replaces it by a new prior point
  with L > L*.
- After *i* replacements the prior volume with L > L* has shrunk to about X_i = exp(−i/n).
- Z = Σ L_i (X_{i−1} − X_i).
- The removed points, weighted by L_i (X_{i−1} − X_i) / Z, are the posterior samples.

The hard part is drawing the new point from the prior restricted to L > L*. For Pandora's 15–17
parameters this is done with Markov chains.

The sampler: ``pandoramoon.nss``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nss`` (nested slice sampler) is written for very cheap likelihoods. A Pandora likelihood call takes
2–10 µs, so a sampler with Python code per call (tens of µs) would spend most of its time on itself.
Everything in one run is compiled with numba: there is no Python per likelihood call and none per
iteration. The sampler's own work costs about 0.25 µs per call, against 2–3 µs for the likelihood.

**One run.**

- n live points (1000 for ``accuracy="high"``, 500 for ``"medium"``) in the unit cube.
- A new point comes from a slice-sampling chain of ``nsteps`` moves at the fixed threshold L*, started
  at a random surviving live point (Neal 2003):

  - pick a direction;
  - place an interval around the point at random;
  - step it out until both ends are below L*;
  - shrink it until a point above L* is found.

  The interval width adapts so that it needs about one step-out per move.
- **Directions ("pairs"):** the difference of a random live point and one of its four nearest live
  neighbours (in coordinates scaled by the live points' spread). This gives the local scale and
  orientation of the constrained region, also for strongly correlated and curved posteriors. The
  direction never depends on the chain's own position, so every move leaves the constrained prior
  invariant (an exact sampler). The neighbour lists are refreshed every n/4 replacements.
- The run stops when the remaining live points can add less than 1 % to Z.

**Many runs, merged.** ``nss.fit``:

- runs many such runs independently, one per core, without synchronisation (near 100 % parallel
  efficiency);
- merges the runs as one run with the summed number of live points (Higson et al. 2018). The merged
  ln Z is about the mean of the runs' ln Z and unbiased in known-evidence tests;
- takes the error of ln Z from the scatter between the runs. On Pandora posteriors this is 1.2–2.3× the
  textbook error sqrt(H/n) of a single run, because of incomplete mixing; the empirical error covers
  this;
- excludes a run that never found the best mode (much lower maximum likelihood and ln Z) and counts it.
  More than 10 % such runs fail the fit.

Self-checks
~~~~~~~~~~~

Slice chains that are too short leave new points correlated with their start points, and this biases
ln Z upwards. Neither the scatter between runs nor the insertion-index test detects it. ``nss.fit``
therefore runs two arms:

- the result arm with chains of ``nsteps`` moves;
- a check arm with ``nsteps / 2``, at about one third of the cost.

It adds runs until all of these hold:

- the error of ln Z ≤ ``lz_tol`` (0.1 for ``"high"``, 0.2 for ``"medium"``);
- the posterior is reproducible: the largest KS distance over all parameters between the merged
  posteriors of the even- and odd-numbered runs is ≤ ``ks_tol`` (0.05 / 0.08);
- the chain length is sufficient: ln Z of the two arms agrees within 2 ``lz_tol`` (including its
  error), and their posteriors within ``ks_tol``;
- at least ``min_runs`` runs (16 / 8) and at most 10 % failed runs.

If the two arms disagree significantly, ``nsteps`` is doubled: the old result arm becomes the check arm
and sampling continues. Data that need longer chains are detected and handled automatically. The checks
are evaluated after every finished run, as soon as a pass is possible, and the fit stops at once. A fit
that does not pass within ``max_runs`` runs is returned with ``passed=False``.

**What the error means.** ``logzerr`` is the statistical error of this fit; ten C0 fits with different
seeds scattered by 0.06, against a quoted 0.07. Different sampler settings (live points, chain length)
can shift ln Z by up to ~0.1. Add this as a systematic error when comparing evidences at the 0.1-nat
level.

Moon orbit: phase-at-transit coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The problem.** A moon is observed only during the planet's transits, once per planetary orbit. Many
moon periods P_m therefore put the moon at nearly the same phases in every transit. In the native
parameters (per_moon, tau_moon) the posterior is a comb of 50–80 very narrow, separated modes, spaced
by about P_m² / P_b in period. No general-purpose sampler mixes between them reliably: evidences
scattered by 1–3 nats between runs and posteriors were not reproducible.

**The coordinates.** ``NestedSampler`` samples instead:

- φ_c: the moon's orbital phase at a reference transit e_c (the mean of the transits in the data);
- δ: the fractional phase advance per planetary orbit;
- m: the integer number of moon orbits per planetary orbit (sampled through a smooth step map u_m → m).

These satisfy P_m = P_b / (m + δ) and φ_e = φ_c + (e − e_c) δ for transit e.

- The moon's phase at every transit depends only on (φ_c, δ). The comb collapses into one broad
  solution in (φ_c, δ) and a smooth sequence in m.
- The map from (φ_c, δ, m) to (per_moon, tau_moon) is a bijection.
- An exact importance weight makes the prior, the evidence and the posterior identical to those of the
  native parameters with your prior.
- Any per_moon prior (Uniform, LogUniform, Normal) is applied as an exact density ratio.
- Eccentric moons work the same way (tau_moon is a mean-anomaly offset in Pandora).
- Results are always reported in the native parameters.

Conditions (``moon_phase="auto"`` checks them):

- the per_moon prior has a finite upper bound ≤ the shortest planet period in the per_bary prior;
- tau_moon ~ Uniform(0, 1).

**Ω double cover.** With Omega_moon ~ Uniform(lo, lo + 180), the node is sampled on [lo, lo + 360). The
orbit at Ω + 180° with the phase shifted by half a turn is the same orbit, so the prior is unchanged.
The sampler no longer sees a boundary at lo and lo + 180. Samples are reported folded back to
[lo, lo + 180).

**Short periods.** A per_moon prior that starts at 0 includes periods so short that the moon would be
inside the planet. Pandora treats those moons as unphysical and returns the planet-only light curve.
Periods below 0.6 × the shortest physical period of the prior box are not sampled. Their prior mass F
(10⁻³ for the notebook's priors) contributes exactly F · Z_planet, which is added back after the
planet-only fit.

Compilation
~~~~~~~~~~~

The likelihood that the sampler sees is a compiled function of the unit-cube point and a tuple holding
the data and all prior constants. It is compiled once per Python process (about 15 s) and reused for
every model, data set and prior, so later fits in the same process start immediately.


Validation
----------

The sampler and the interface were validated on the server used for the development (16 cores),
see ``SAMPLING.md`` and ``NSS_SAMPLER.md`` in the project directory. In summary:

- **Known evidences:** Gaussian mixtures and a 16-dimensional problem with bananas, prior-edge
  truncation and strong correlations. Unbiased within ±0.05 once the chains are long enough. The
  chain-length check bounds the remaining bias to ~0.1 nats.
- **A suite of 12 injected data sets:** more or fewer transits, noise from 100 to 2000 ppm, 2-min
  cadence, no moon, a long-period inclined moon, a missing transit, a Kepler-like planet, and the
  Pandora paper's system. Every case passes its checks by itself, and the best fit lies above the
  injected truth in every case. All 12 together take 55 minutes on 16 cores.
- **Simulation-based calibration:** 263 planet+moon fits and 256 planet-only fits of injections drawn
  from the prior (``accuracy="medium"``). The rank of the true value in its posterior is uniform for
  all 15 resp. 7 parameters, so the posteriors have the right width and location.
- **The interface itself:**

  - the compiled likelihood equals ``Likelihood`` to 10⁻¹⁴;
  - prior transforms, the moon-phase weights and the limb-darkening prior reproduce their target
    distributions;
  - resuming after a crash gives exactly the uninterrupted result;
  - evidences agree with the earlier scripts (notebook data: −185.975 ± 0.018 over five seeds vs
    −185.95 ± 0.02) and with an independent prediction for an informative limb-darkening prior.
  - Tests: ``test_cases/test_sampling.py``.


Limitations
-----------

- The noise model is white Gaussian noise with the given errors. There is no jitter term and no
  correlated noise.
- Supersampling (long-cadence data) is supported through the ``Likelihood``, but its effect on fits has
  not been tested yet.
- The chain-length check decides with fixed thresholds. Near them, a fit may double its chains
  unnecessarily (costing time, not accuracy).
- Very narrow per_moon priors are handled through a density ratio. They are correct but sample less
  efficiently.
- One process at a time should use the cores (``threads``): fits are already parallel.


References
----------

- Skilling 2006, Bayesian Analysis 1, 833 (nested sampling)
- Neal 2003, Annals of Statistics 31, 705 (slice sampling)
- Higson, Handley, Hobson & Lasenby 2019, Statistics and Computing, arXiv:1704.03459 (merging runs, errors)
- Fowlie, Handley & Su 2020, MNRAS 497, 5256 (insertion-index test)
- Karamanis & Beutler 2021, Statistics and Computing, arXiv:2002.06212 (ensemble slice sampling)
- Kipping 2013, MNRAS 435, 2152 (limb-darkening parametrisation)
- Kass & Raftery 1995, JASA 90, 773 (Bayes factors)
- Talts et al. 2018, arXiv:1804.06788 (simulation-based calibration)
- Buchner 2021, JOSS 6, 3001 (UltraNest; result conventions)
- Hippke & Heller 2022, A&A, arXiv:2205.09410 (Pandora)
