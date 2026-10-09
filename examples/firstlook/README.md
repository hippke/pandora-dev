# First-look exomoon screening (example)

A cheap first pass over many transiting-planet light curves. It flags the ones that deserve a full
planet+moon retrieval. It is not part of the `pandoramoon` core: it is an example built on
`pandoramoon.nss` (compiled nested sampler) and `pandoramoon.reparam` (phase-at-transit moon
coordinates), for users to adapt.

## What it does

For one light curve:
1. **Planet-only model:** one short nested-sampling run (7 parameters).
2. **Planet+moon model:** one short nested-sampling run (16 parameters, moon orbit in
   phase-at-transit coordinates).
3. **Output:**
   - the log Bayes factor `dlnz = lnZ(planet+moon) − lnZ(planet only)`;
   - the verdict `interesting = dlnz > threshold`;
   - a rough summary of the moon (period, radius, mass ratio, inclination; median and 16/84 %).

Each model is one run with few live points and short slice chains, without the convergence checks
of `nss.fit`. The evidences are therefore noisy (±0.5) and slightly biased, which is harmless far from
the threshold. **Interesting light curves must be followed up with a full fit** (`nss.fit`, ~10⁹
calls).

```python
import firstlook
fl = firstlook.FirstLook(time, flux, flux_err, t0_bary=11.0, per_bary=(360.25, 370.25), a_bary=(210, 220))
r = fl.run(threshold=...)          # one light curve on one core
print(r["dlnz"], r["interesting"], r["moon"])

results = firstlook.run_many(list_of_firstlooks, threads=16)   # many light curves, one per core
```

All compiled functions take the light curve as an argument. numba compiles once per process
(~30 s), not per light curve.

## Files

| file | what |
|---|---|
| `firstlook.py` | `FirstLook` (one light curve), `run_many`, the compiled likelihoods `ll_planet`, `ll_moon` |
| `injections.py` | white-noise injections: Kepler-like (30 min, 3–9 transits) and PLATO-like (10 min, 2–3 transits) planets with and without moons |
| `bench_firstlook.py` | calibration (reference runs + first-look settings on the same injections, `--summary`) and the **speed benchmark** (`--bench`) |

## Speed benchmark

Run it after any optimisation of Pandora or nss:

```bash
python bench_firstlook.py --bench --labels cal.jsonl --n-moon 60 --n-null 40 --out /some/dir/bench.jsonl
```

It runs the default first look on a fixed set of injections and appends one line to
`bench_log.jsonl` (next to `--out`):
- CPU seconds per light curve, light curves per hour per core, and calls per light curve;
- the threshold for 2 % false passes, and the recall of well-detectable moons (reference
  dlnZ ≥ 8, from `--labels`).

A speed-up only counts if the recall stays the same.

## Calibration and status

Status 2026-10-09 (white noise, known noise level, injections from `injections.py`; details in the
project's SAMPLING.md, Round 11). Default first look: one run per model with 200 live points (nsteps
28 / 32), threshold dlnZ > 2. "Well detectable" = a reference run with 1000 live points gives
dlnZ ≥ 8. Two injection sets:
- standard: noise 100–1000 ppm, 400 light curves;
- faint: 600–4000 ppm, 280 light curves.

| | standard | faint | both |
|---|---|---|---|
| recall, reference dlnZ ≥ 8 | 166 / 166 | 66 / 66 | **232 / 232 (100 %; 95 % lower limit 98.7 %)** |
| recall, reference dlnZ 5–8 | 3 / 3 | 5 / 5 | 8 / 8 |
| recall, reference dlnZ 2–5 | 6 / 8 | 5 / 5 | 11 / 13 |
| no-moon light curves passed | 0 / 160 | 1 / 80 (dlnZ 2.03; the reference run gives 2.27) | **1 / 240 (0.4 %)** |
| CPU-s per light curve | 6.4 | 4.6 | |
| likelihood calls per light curve | 2.0 M | 1.4 M | |

- The moon parameters are rough but honest. For detected moons, the truth lies inside the 16–84 %
  interval in 64–68 % of the cases. Median errors: P_m 18 %, r_m 4 %, M_m/M_p 26 %.
- Speed benchmark baseline: 6.2 CPU-s per light curve, 578 light curves per hour per core (AMD
  Ryzen 9 7950X).
- A cheaper setting with 100 live points and nsteps 16 (1.2–1.7 CPU-s) misses 1–2 % of the
  well-detectable moons: with few live points the run sometimes never finds the moon.

Not tested yet:
- red noise, stellar variability and detrending residuals (expect more false passes);
- unknown noise levels;
- other planet / star types;
- eccentric orbits;
- single transits.

