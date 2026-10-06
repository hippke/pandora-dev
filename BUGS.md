# Pandora Bug Audit

Date: 2026-10-06
Base commit: `baa7788` ("Update README.md")
Scope: full review of `pandoramoon/` (`pandora.py`, `occult.py`, `eclipse.py`, `ellipse.py`, `grids.py`, `helpers.py`), examples, and notebooks. Every bug below was verified numerically against reference implementations before fixing; short repro snippets are given. Line numbers refer to the base commit (they shift after fixes).

Status legend:
- **FIXED** - fixed in this patch series (commit subject given)
- **OPEN** - known, deliberately not fixed yet (details below)
- **IGNORED** - accepted for now per project decision

---

## 1. `resample()` drops the last supersampled point of every cadence - FIXED

Severity: critical (wrong results whenever `supersampling_factor > 1`)
Location: `pandoramoon/helpers.py:8-13`
Fixed in: "Fix off-by-one in resample() supersampling average (bug #1)"

```python
end = start + factor - 1
out_arr[idx] = np.mean(arr[start:end])
```

`np.mean(arr[start:end])` excludes the end index, so only `factor - 1` of the
`factor` sub-cadences are averaged; the last sample of every cadence is
silently discarded. With `supersampling_factor = 2` no averaging happens at
all (plain decimation). Verified: `resample([0..9], 5)` returned
`[1.5, 6.5]` instead of `[2.0, 7.0]`.

The fix averages `[start, start + factor)` and additionally includes a
partial trailing group when `len(arr) % factor != 0` (previously the tail was
dropped - see also #17) so no data is silently lost.

## 2. `ellipse_ecc()`: `if M.any() > pi:` never True - eccentric moon positions wrong for half of every orbit - OPEN (not fixed yet)

Severity: critical for eccentric moon orbits (`e_moon > 0`)
Location: `pandoramoon/ellipse.py:36-41` and `54-55`

The code:

```python
M = (2 * pi / per) * (time - (tau * per))
flip = False
M = M - (np.floor(M / (2 * pi)) * 2 * pi)
if M.any() > pi:
    M = 2 * pi - M
    flip = True
...
if flip:
    k = 2 * pi - k
```

**Intent.** The Kepler starter + high-order corrector used here (lines 42-53)
is accurate for mean anomaly `M` in `[0, pi]`. For `M` in `(pi, 2*pi)` the
standard technique is to solve with the reflected anomaly `2*pi - M` and
reflect the result back, using the symmetry
`E(2*pi - M) = 2*pi - E(M)`. The `flip` flag implements this reflection: it is
set when any part of the orbit is in the second half, the solver then runs on
the reflected anomaly, and the eccentric anomaly `k` is un-reflected at the
end.

**What actually happens.** `M` is a 1-D array (one mean anomaly per time
stamp). `M.any()` reduces the whole array to a *single* boolean ("is any
element nonzero") - it is not an element-wise comparison. The condition
therefore evaluates `True > pi` (or `False > pi`), i.e. `1 > 3.14159...`,
which is always `False`. The reflection branch is dead code: `flip` is always
`False`, `M` is never reflected, and `k` is never un-reflected. This was
confirmed by static analysis and by matching the numba function's output
against a pure-Python replication of the same (unreflected) code path.

**Measured impact.** Comparing the code's eccentric anomaly `k` against a
bisection reference solver (converged to machine precision) over a full
orbit (20,000 mean anomalies; numba replica with the same `fastmath=True`
flags verified identical to the shipped function to 7e-16):

| e    | max error for M in [0, pi) | max error for M in [pi, 2*pi) |
|------|----------------------------|-------------------------------|
| 0.2  | 8.9e-16                    | 4.6e-07                       |
| 0.5  | 8.9e-16                    | 5.6e-04                       |
| 0.9  | 8.9e-16                    | 1.9e-01                       |

The first half of the orbit is machine-precision accurate. On the second
half the error is not uniform: it stays near machine precision until
roughly M ~ 4.8 (~60% into the second half) and then grows rapidly toward
M -> 2*pi, i.e. as the moon approaches the *next* periastron passage
(the starter formula `alpha ~ (pi - abs(M))` is progressively further
outside its design range as |M| exceeds pi, and the fixed-order g/h/k
corrector amplifies rather than fully corrects a poor starter). At e = 0.9
the anomaly error reaches ~0.18 rad near M -> 2*pi. The corresponding sky
position error is muted near periastron (the orbit is small and slow-moving
in radius there): measured against a correctly scaled reference, the worst
sky-position error at e = 0.9 is ~1.7% of a_moon (mean ~0.1% over the
second half), concentrated in the final ~15% of the orbital period.

What breaks is the *time parameterization*, not the orbit geometry: the
code's radius `1 - e*cos(k)` and true anomaly `nu(k)` both derive from the
same (wrong) `k`, so the moon always lies on the correct Keplerian ellipse
- but at the wrong point of it for the given time stamp. The anomaly offset
delta(M) varies from ~0 at M ~ pi to ~0.18 rad at M -> 2*pi (e = 0.9), so
it cannot be absorbed by any fixed redefinition of `tau_moon` or
`w_moon`; the model family cannot represent the true Keplerian ephemeris
for that part of the orbit.

**Second, structural problem.** Even if the comparison were fixed to
`(M > pi).any()`, a single scalar `flip` cannot be correct for a *time array*:
real grids (e.g. `per_moon = 0.3 d` over a `0.6 d` transit window) contain time
stamps in both halves of the orbit. The reflection must be applied per
element: for each time stamp either solve for `M` and use `E`, or solve for
`2*pi - M` and use `2*pi - E`.

**Consequences.**
- Any model with `e_moon > 0` places the moon at the wrong point of its
  (correctly shaped) orbit for the time stamps in the affected zone near the
  periastron recurrence; the anomaly error grows rapidly with eccentricity
  and with proximity to `M -> 2*pi`.
- Injection-retrieval loops that use the same code for injection and
  retrieval partially absorb the error into other parameters, but because
  delta(M) varies across the orbit it cannot be fully absorbed: the
  `e_moon` / `w_moon` / `tau_moon` posteriors are biased, and the model
  cannot represent the true Keplerian ephemeris near the periastron
  recurrence.
- Circular moon orbits (`e_moon = 0`) use `ellipse()` instead and are
  unaffected.
- Interplay with bug #8 (fixed): the former `r = -(1 - e*cos(k))` negation
  mirrored the whole orbit through the barycenter, equivalent to redefining
  `w_moon -> w_moon + 180 deg`. With #8 fixed, the `M in [0, pi)` half now
  follows the standard convention; the `M >= pi` half remains incorrect until
  this bug is fixed.

**Why not fixed yet (project decision).** The correct fix requires an
element-wise restructure of the solver (loop over time stamps or vectorized
mask logic), which changes the numerical output for every eccentric-moon
model and invalidates comparison with previously generated injection data.
It should be accompanied by (a) a regression test against an independent
reference solver (e.g. `scipy.optimize.brentq`/Newton to 1e-12) over a dense
`(M, e)` grid, and (b) a check that example notebooks still recover injected
parameters.

**Planned fix sketch.**

```python
for idx in range(len(time)):
    m = (2 * pi / per) * (time[idx] - tau * per)
    m = m - (np.floor(m / (2 * pi)) * 2 * pi)
    flip = m > pi          # per element, not M.any() > pi
    if flip:
        m = 2 * pi - m
    ... starter + corrector for m -> k ...
    if flip:
        k = 2 * pi - k
```

(or fully vectorized with `np.where`). Runtime cost is expected to be small
relative to the rest of the model.

## 3. `read_occult_cache()`: index wraparound for k <= 0.001 erases transits - FIXED

Severity: critical (silent signal loss for very small bodies)
Location: `pandoramoon/occult.py:26-27`; call sites `pandoramoon/pandora.py:438-443, 451-452`
Fixed in: "Fix occult-cache wraparound for k below grid; route to occult_small (bug #3)"

```python
idx_k = int(np.ceil((k - ks[0]) / (ks[1] - ks[0])))
```

For `k <= ks[0] = 0.001` this yields `idx_k = 0`, and the subsequent
`fs[idx_k - 1]` / `ks[idx_k - 1]` wrap around to the *last* cache row
(k = 0.1) with a negative interpolation weight. Combined with the
`flux > 1` clamp, the light curve of any body with `k < 0.001` is clipped to
1: the transit disappears entirely. Verified end-to-end via
`moon_model.light_curve`: `r_planet = 0.0005` and `0.0009` produced depth
exactly 0.0 (expected ~3.0e-7 and ~9.8e-7, i.e. k^2 / (1 - u1/3 - u2/6));
`k >= 0.001` is accurate.

Trigger paths: any `r_planet < 0.001` (cache is always used for
`r_planet < 0.1`), or `r_moon < 0.001` when `occult_small_threshold < 0.001`.

The fix clamps `idx_k` into `[1, len(ks) - 1]` in `read_occult_cache()` and,
in `pandora()`, routes bodies with `k` below the cache grid (`cache[1][0]`)
to `occult_small()`, which is exact to < 0.1 ppm there.

## 4. `z += 1e-6` guard can round *back onto* k -> ZeroDivisionError - FIXED

Severity: high (latent crash of the whole sampler)
Location: `pandoramoon/occult.py:220-221` (`occult`), `383-384` (`occult_hybrid`)
Fixed in: "Fix z-k perturbation rounding onto k (crash) (bug #4)"

```python
if abs(z - k) < 1e-6:
    z += 1e-6
```

For example `z = 0.299999`, `k = 0.3`: the float64 difference is
`9.999999999e-07 < 1e-6`, so the guard fires - and `0.299999 + 1e-6` rounds to
exactly `k`. Then `x1 = (k - z)**2 = 0`, and the case-III/IV formulas divide
by `x1` (`3*x3/x1`, `x2/x1`) -> `ZeroDivisionError` (numba raises for float
division by zero). Verified: `occult([0.299999], 0.3, ...)` and
`occult_hybrid([0.299999], 0.3, ...)` both crash; the crash set includes
k = 0.26...0.5 (float-rounding dependent) at `z = k - 1e-6`.

Per-call probability is tiny, but a long nested-sampling run evaluates
billions of (z, k) combinations and would eventually die. The fix assigns
`z = k + 1e-6` instead of incrementing, which can never round onto `k`.

## 5. Case-IV `+1e-8` regularizer pushes q above 1 -> ZeroDivisionError just below z = 1 - k - FIXED

Severity: high (latent crash in a narrow z-band)
Location: `pandoramoon/occult.py:316`
Fixed in: "Remove case-IV q regularizer that pushed q above 1 (crash band) (bug #5)"

```python
q = sqrt((x2 - x1) / (1 - x1) + 1e-8)  # re-calc because different condition
```

In case IV (`z < 1 - k`, k < 0.5), the ratio `(x2 - x1)/(1 - x1)` approaches 1
as `z -> 1 - k` from below. The `+ 1e-8` pushes `q` slightly above 1 for z
within ~1e-9 of `1 - k`, so `kc = sqrt(1 - q**2)` receives a negative
argument, NaN-poisons `ellpicb`, and numba raises `ZeroDivisionError`.
Verified: `occult([0.9 - 1e-9], 0.1, ...)` crashes while `0.9 - 1e-8` is fine;
band width grows with k (~1e-9 at k = 0.05, ~5e-9 at k = 0.3). The same
mechanism crashes `occult(0.3, 0.7)` where float rounding routes the
boundary point into case IV. In pure Python the failure mode is a log domain
error instead. Any point in the affected band lies deep inside the
`abs(k + z - 1) < 1e-4` special-case band, where the general (q-based) value
is immediately overwritten - the regularizer serves no purpose.

The fix removes the `+ 1e-8` (q is mathematically < 1 inside case IV, and
q = 0 at z = 0 is handled fine by `ellk`/`ellec`/`ellpicb`) and adds a
defensive `q >= 1 -> q = 1 - 1e-8` clamp for float-rounding edge cases.

A follow-up commit applies the same clamp to `occult_hybrid`'s case IV,
which had no guard at all: at the `z = 1 - k` boundary float rounding can
make `q` *exactly* 1.0 (e.g. z = 0.3, k = 0.7), so `kc = sqrt(1 - q**2) = 0`
and the `ellpicb` iteration can never converge. Before bug #18's fix this
silently returned 0 after 1000 dead iterations (harmless only because the
`|k + z - 1| < 1e-4` special case overwrites `ldx` there); with #18 it
raised. `ellpicb` itself converges for any q < 1 (verified down to
1 - q ~ 3e-16), so clamping q below 1 is sufficient. Verified with a dense
boundary sweep (k = 0.02...0.98, z within 1e-9...1e-3 of 1 - k and of k):
no exceptions, both functions agree with the reference.

## 6. Boundary gaps at z = 1 - k / z = k leave `ld`/`ed` uninitialized -> garbage flux or crash - FIXED

Severity: high (silently wrong flux at exact boundary values)
Location: `pandoramoon/occult.py:282-284` (`occult` case III, second clause),
`408-409` (`occult_hybrid` case III, both clauses)
Fixed in: "Close z=1-k boundary gaps in occult/occult_hybrid; add missing q guard (bug #6)"

`occult_hybrid` uses strict inequalities in both case-III clauses
(`z > 0.5 + abs(k - 0.5)` and `z > abs(1 - k)`), while case IV uses strict
`z < 1 - k`. The single points `z = 1 - k` (k <= 0.5) and `z in {1 - k, k}`
(k > 0.5) are therefore covered by *no* branch: `ldx`/`edx` keep their
`np.empty` contents and the returned flux is arbitrary. Verified:
`occult_hybrid([0.9], 0.1)` returned 0.989231 where the correct value is
0.991853; `occult_hybrid([0.4], 0.6)` returned 0.6957 vs 0.6080. The file's
own header comment notes the `>=` fix ("must be >= in z >= 0.5 + abs(k - 0.5)")
- but it was applied only to `occult`, not to `occult_hybrid`.

`occult` has the same gap at `z = abs(1 - k)` for k > 0.5 (strict `z >` in the
second clause): the boundary point either falls through to uninitialized
values or, due to float rounding of `1 - k`, enters case IV and crashes via
bug #5. Verified: `occult([0.3], 0.7)` -> ZeroDivisionError; `occult([0.4], 0.6)`
uses stale memory.

The fix makes the case-III lower bounds inclusive in both functions
(matching the Mandel & Agol 2002 validity range `|1 - k| <= z <= 1 + k`), adds
the missing q-regularization guard (`q -> 1 - 1e-8` when q is within 1e-8 of
1, as already present in `occult`) to `occult_hybrid` case III, and clamps the
`sqrt((1 - x1) * (x2 - 1))` argument in case III with `max(..., 0)` to guard
against float-rounding negatives at the boundary.

## 7. `occult_hybrid` writes to `flux[-2]`/`flux[-1]` and reads uninitialized memory - FIXED

Severity: medium (undefined behavior, harmless only by accident)
Location: `pandoramoon/occult.py:386-389, 485-487`
Fixed in: "Guard negative-index flux writes in occult_hybrid (bug #7)"

The loop `for i in range(-2, len(zs))` uses `i = -2, -1` for the two
interpolation support points (z = 0 and z = 0.65) which do not correspond to
array elements. The code nonetheless executes `flux[i] = 1` (unocculted
branch) and reads/writes `flux[i]` in the final clamp. With Python/numba
negative indexing these writes hit the *last two elements* (later overwritten
- benign), and for `len(zs) < 2` they are out-of-bounds (undefined behavior
under numba, which does not bounds-check by default). The fix guards all
`flux[i]` accesses with `i >= 0`.

## 8. `ellipse_ecc()` negates the radius: `r = -(1 - e*cos(k))` - w_moon convention off by 180 deg - FIXED

Severity: medium (wrong parameter interpretation; inconsistent conventions)
Location: `pandoramoon/ellipse.py:56`
Fixed in: "Fix w_moon convention: remove radius negation in ellipse_ecc (bug #8)"

The negative radius mirrors both the moon and the planet through the
barycenter, which is equivalent to redefining `w_moon -> w_moon + 180 deg`
(the traced orbit is still a valid Kepler orbit with that flipped periastron,
for the M < pi half - see bug #2 for the other half). Verified: at
perisastron passage time `t = tau_moon * per_moon` the code placed the moon at
position angle 180 deg instead of the given `w_moon = 20 deg`. The convention
was also inconsistent with the circular model `ellipse()`, where the same
`tau_moon` puts the moon on the opposite side.

Because `w_moon` (and `tau_moon`) are free parameters in retrievals, fits are
self-consistent either way, but the parameter meaning differs from the
documented `[deg]` convention and from the e = 0 model. The fix uses the
positive radius. Note: this changes model output for all `e_moon > 0`
injections (the moon now appears where the parameters say it should).

## 9. Examples use `occult_small` far beyond its validity range - IGNORED (per project decision)

Severity: medium (model error larger than injected noise in the examples)
Location: `examples/*.py` (`params.occult_small_threshold = 0.1`),
`pandoramoon/occult.py:70-71` (docstring: "Adequate for k<~0.01")

The small-planet approximation assumes the occulted stellar patch is
uniformly lit. Measured error against the exact Mandel-Agol model
(u1 = 0.4089, u2 = 0.2556):

| k       | max error        |
|---------|------------------|
| 0.01    | 4.3 ppm          |
| 0.02    | 25.9 ppm         |
| 0.03526 | 105.4 ppm        |
| 0.05    | 248.1 ppm        |
| 0.1     | 1335.5 ppm       |

All examples set `occult_small_threshold = 0.1` with `r_moon = 0.03526`, so
the moon light curve carries ~105 ppm model error - larger than the 100 ppm
injected noise. The default in `model_params` (0.01, ~4.3 ppm at the
boundary) is consistent with the docstring; the examples' 0.1 is not.
Accepted for now; consider lowering the examples' threshold to 0.01 or
routing through the cache instead.

## 10. `x_bary_grid()` epoch rounding truncates toward zero for data before t0 - FIXED

Severity: low/medium (only when data exist before `t0_bary`)
Location: `pandoramoon/grids.py:57`
Fixed in: "Fix epoch rounding for data before t0 (bug #10)"

```python
epoch = int((time[idx] - t0_bary) / per_bary + 0.5)
```

Python/numba `int()` truncates toward zero, so negative values in
`(-1, -0.5]` round to 0 instead of -1. Any epoch more than half a period
*before* `t0_bary` gets the wrong epoch index and a wildly wrong position
(verified: `x_bary ~ -1400` instead of ~+1 for a time one period before
t0). The fix uses `np.floor`.

## 11. `occult_hybrid` interpolation branch: dead `testv`, duplicate expensive call, missing clamp - FIXED

Severity: low (performance + code quality; missing clamp is cosmetic)
Location: `pandoramoon/occult.py:445-451`
Fixed in: "Remove dead testv and duplicate call; clamp interpolated flux (bug #11)"

```python
testv = occult_small_single_value(z, k, u1, u2)   # never used
flux[i] = (
    occult_small_single_value(z, k, u1, u2)        # computed a second time
    + interpol_flux_1 + interpol_flux_2 * z)
continue                                          # skips the flux > 1 clamp
```

The interpolation branch calls the (non-trivial) small-planet approximation
twice per point and discards the first result, and its result bypasses the
`flux > 1` clamp that all other branches apply. The fix computes it once and
clamps.

## 12. `occult` case-III q guard fires for every q - FIXED

Severity: low (numerically negligible, but wrong logic)
Location: `pandoramoon/occult.py:287-289`
Fixed in: "Fix case-III q guard condition in occult (bug #12)"

```python
if (q - 1) < 1e-8:
    q -= 1e-8
```

The intent is to push q away from the ellk singularity at q = 1, i.e. the
condition should be `1 - q < 1e-8`. As written, `(q - 1) < 1e-8` is true for
*every* q < 1 + 1e-8, so all case-III evaluations get their q shifted by
-1e-8. The induced flux error is O(1e-8) except within ~1e-8 of q = 1, where
the shift actually saves the evaluation - which is why the bug went
unnoticed. The fix corrects the condition; the affected q values still get
shifted.

## 13. `create_occult_cache()` ignores its own `k_min` - FIXED

Severity: low (dead constant, hardcoded duplicate)
Location: `pandoramoon/occult.py:11, 14`
Fixed in: "Use k_min constant in create_occult_cache grid (bug #13)"

`k_min = 0.001` is defined but the grid uses a hardcoded
`np.linspace(0.001, k_max, dim)`. The fix uses the constant (no behavior
change). Note the cache itself stores float32, quantizing fluxes to ~6e-8
(~0.06 ppm) - adequate for Kepler-class photometry, marginal for anything
much more precise.

## 14. `timegrid()`: duplicated line; invalid `supersampling_factor` only prints - FIXED

Severity: low
Location: `pandoramoon/grids.py:25-28`
Fixed in: "Validate supersampling_factor; remove duplicated line (bug #14)"

`supersampled_cadences_per_day` is assigned twice verbatim (refactoring
leftover), and a `supersampling_factor < 1` only prints a message before
silently building an empty/degenerate time grid. The fix removes the
duplicate and raises `ValueError`.

## 15. `ld_invert()` divides by zero for u1 + u2 == 0 - FIXED

Severity: low (edge case of a helper)
Location: `pandoramoon/helpers.py:24-26`
Fixed in: "Guard ld_invert against u1+u2 == 0 (bug #15)"

`q2 = u1 / (2 * (u1 + u2))` raises `ZeroDivisionError` for the degenerate
Kipping case (u1, u2) = (0, 0). The fix returns `(0.0, 0.5)`, which maps back
to (0, 0) under `ld_convert` (any q2 does once q1 = 0).

## 16. `occult_hybrid` leaves `le/ld/ed` uninitialized for k > 1, z < k - 1 - FIXED

Severity: low (unphysical k > 1, but pandora() has no guard against it)
Location: `pandoramoon/occult.py:373-489` (missing complete-occultation branch)
Fixed in: "Handle complete occultation (k >= 1) in occult_hybrid (bug #16)"

`occult` returns flux 0 for `k >= 1 and z <= k - 1` (star fully behind the
occulter); `occult_hybrid` has no such branch, so for k > 1 none of the
lex/ldx/edx cases fire and the flux is uninitialized memory. Verified:
`occult_hybrid([0.0, 0.05, 0.1, 0.19], 1.2)` returned 0.2217 for all points
(stale memory) where the correct value is 0. `pandora()` does not currently
guard `r_planet >= 1` or `r_moon >= 1`, so this path is reachable with
nonsense parameters. The fix adds the same complete-occultation branch as
`occult`.

## 17. `resample()` drops the trailing partial group - FIXED (with #1)

Severity: low
Location: `pandoramoon/helpers.py:8`
Fixed in: "Fix off-by-one in resample() supersampling average (bug #1)"

`int(len(arr) / factor)` truncates, so a trailing group shorter than
`factor` was silently discarded (e.g. cadences = 98 with factor 5 loses the
last 3 supersamples). The fix includes the partial group in the output.

## 18. `ellpicb()` silently returns 0 on non-convergence - FIXED (eclipse limitation documented)

Severity: low
Location: `pandoramoon/occult.py:125-138`
Fixed in: "Raise on ellpicb non-convergence instead of silent 0 (bug #18)"

If the Bulirsch iteration does not converge within 1000 rounds the function
returns 0, which flows into the limb-darkening terms as a silently wrong
value. In practice convergence takes < 10 iterations, but the failure mode
should be loud. The fix raises `ValueError`.

## 19. Examples set `params.e_moon`, but the model reads `params.ecc_moon` - OPEN (fix together with #2)

Severity: medium (examples silently ignore the moon eccentricity)
Location: `examples/example.py:29`, `examples/example_figure.py:29`,
`examples/example_video.py:29`, `examples/example.ipynb`,
`examples/injection_retrieval_simple_ultranest.ipynb`; attribute defined as
`ecc_moon` in `pandoramoon/pandora.py:42` and read in `moon_model.__init__`.

Discovered during final verification of this patch series: the standard
example output was bit-identical before and after fixing bug #8, although it
claims `e_moon = 0.9` and should have exercised `ellipse_ecc()`. The reason:
all examples assign `params.e_moon = 0.9`, but `model_params` defines the
attribute as `ecc_moon` (default 0) and `moon_model` reads `params.ecc_moon`.
The misspelled attribute is silently ignored, so the examples actually model a
*circular* moon orbit via `ellipse()`; the intended eccentric case never runs.
(The injection-retrieval notebook uses `e_moon = 0`, which coincides with the
effective default, so its injection is unaffected.)

Not fixed in this series on purpose: renaming the attribute would make the
examples exercise `ellipse_ecc()` with e = 0.9, i.e. the still-open bug #2
(broken M > pi reflection), and the tutorial video would show a
non-Keplerian moon for half of each orbit. Fix `examples/*.py` and both
notebooks (`e_moon` -> `ecc_moon`) together with bug #2, then verify that
injection-retrieval recovers the injected parameters.

## Known limitations (documented, not bugs)

- `eclipse()` models only "planet occults moon" (the moon behind the planet).
  When the moon is in front of the planet their shadows overlap on the star;
  the total flux then double-counts the overlap. This is the documented
  approximation of Pandora (see docstring and the paper); fixing it requires
  the line-of-sight separation, which the 2D model does not track.
- `pixelart()` mutual-eclipse areas are computed on a
  `numerical_grid`-sized raster (default 25x25), ~2-4% pixelization error on
  partial planet-moon overlaps near the limb (design tradeoff).
- The occultation cache is float32 and bilinear in (z, k): ~0.06 ppm
  quantization plus interpolation error of a few ppm near ingress (measured
  worst case 3.7 ppm over k in [0.002, 0.099]).
- `pandora()` has no guard against `r_planet >= 1` / `r_moon >= 1`
  (now handled inside `occult_hybrid` via bug #16's fix) or negative radii.

## Verification

All fixes were verified with targeted numerical tests (repros above plus
consistency checks `occult` vs `occult_hybrid` vs `occult_small` across
k in [0.001, 0.95] and full z ranges, and a reference Newton Kepler solver),
followed by a full `examples/example.py` run (light curve + 450-frame video).

The model output changes relative to the base commit only for:
- supersampled light curves (bug #1 - previously wrong),
- bodies with k < 0.001 used through the cache (bug #3 - previously erased),
- eccentric moon positions (bug #8 - previously mirrored by 180 deg),
- the exact boundary/crash points (bugs #4-#7, #16 - previously garbage or
  exceptions).

The standard `examples/example.py` run is bit-identical to the base commit
(its planet uses `occult_hybrid`, its moon uses `occult_small` with the
default circular orbit - see bug #19 - and none of its (z, k) values hit the
fixed boundary points), confirming the fixes are non-regressive on the
normal path. Its 450-frame video renders in ~26 s.
Everything else (including the standard example planet light curve) is
unchanged to numerical precision.
