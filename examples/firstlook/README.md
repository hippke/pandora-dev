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
r = fl.run(recall=0.95)            # one light curve on one core (0.90, 0.95, 0.99, 0.999)
print(r["dlnz"], r["interesting"], r["moon"])

results = firstlook.run_many(list_of_firstlooks, threads=16)   # many light curves, one per core
```

All compiled functions take the light curve as an argument. numba compiles once per process
(~30 s), not per light curve.

## Choosing the recall

`fl.run(recall=0.95)` (or 0.90, 0.99, 0.999) picks the cheapest tested setting for the requested
recall of well-detectable moons (reference dlnZ ≥ 8). The knob is not the threshold; it is the
sampler effort:
- a well-detectable moon is missed only when a short moon run never finds the moon's mode;
- that happens in a few % of the runs with 50 live points; no light curve is missed every time
  (worst case 60 %);
- `repeats` independent moon runs, combined as the mean of their Z, all miss with probability
  ~p^repeats.

Several small runs are cheaper than one large one for the same recall. All presets use a planet
run with 100 live points (nsteps 16), moon runs with 50 live points (nsteps 16), and the
threshold dlnZ > 3.

Validated end to end (`recall_study.py --validate`, fresh seeds, the 232 moons with reference
dlnZ ≥ 8 and the 240 no-moon light curves of both injection sets; white noise):

| `recall=` | moon runs (50 live) | measured recall | misses | false passes | CPU-s / light curve | calls | light curves / hour / core |
|---|---|---|---|---|---|---|---|
| 0.90, **0.95 (default)** | 1 | **97.9 %** | 24 / 1160 | 2 / 720 | **1.05** | 0.33 M | ~3400 |
| 0.99 | 3 | **99.78 %** | 4 / 1856 | 2 / 720 | **2.1** | 0.64 M | ~1700 |
| 0.999 | 6 | **99.97 %** | 1 / 3016 | 1 / 720 | **3.7** | 1.1 M | ~1000 |

- 0.90 uses the 0.95 setting; nothing cheaper has been tested.
- The 0.99 and 0.999 rows were measured before the retry safeguard (below) and can only improve
  with it.
- 99.9 % is at the limit of what 3016 trials can show: 1 miss, 95 % upper limit of the miss rate
  0.13 %.
- For comparison, `recall=None` (one moon run with 200 live points) costs 6.4 CPU-s, and a full
  checked fit ~3600 CPU-s.
- **Retry safeguard:** the moon model (almost) contains the planet-only model. So a moon run more
  than 15 nats below the planet-only run has collapsed onto a wrong solution (~0.7 % of the runs
  with 50 live points). Up to 2 extra moon runs are added in that case. This raised the 0.95
  preset from 97.0 % to 97.9 % at unchanged cost.
- The remaining misses are moons a short run finds only sometimes (worst light curve: 60 % miss
  rate per 50-live run). One moon (PLATO-like, reference dlnZ 18.5) is missed by most runs at every
  preset.

## Files

| file | what |
|---|---|
| `firstlook.py` | `FirstLook` (one light curve), `run_many`, the compiled likelihoods `ll_planet`, `ll_moon` |
| `injections.py` | white-noise injections: Kepler-like (30 min, 3–9 transits) and PLATO-like (10 min, 2–3 transits) planets with and without moons |
| `recall_study.py` | recall vs cost: per-light-curve miss rates of several moon-run sizes from repeated seeds (`--analyse`), and end-to-end validation of the presets (`--validate`) |
| `bench_firstlook.py` | calibration (reference runs + first-look settings on the same injections, `--summary`) and the **speed benchmark** (`--bench`) |

## Speed benchmark

Run it after any optimisation of Pandora or nss:

```bash
python bench_firstlook.py --bench --labels cal.jsonl --n-moon 60 --n-null 40 --out /some/dir/bench.jsonl
```

It runs the first look (`--preset 0.95`, default) on a fixed set of 200 injections and appends one line to
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
- Speed benchmark (`--bench`, 200 light curves, AMD Ryzen 9 7950X): preset 0.95 1.05 CPU-s per light
  curve (3437 per hour per core); 0.99 2.0 s (1769); 0.999 3.6 s (1004). The older 200-live setting:
  6.2 s (578).
- The calibration table above is for the older 200-live setting (`recall=None`); the presets are in
  "Choosing the recall".
- A cheaper setting with 100 live points and nsteps 16 (1.2–1.7 CPU-s) misses 1–2 % of the
  well-detectable moons: with few live points the run sometimes never finds the moon.

Not tested yet:
- red noise, stellar variability and detrending residuals (expect more false passes);
- unknown noise levels;
- other planet / star types;
- eccentric orbits;
- single transits.

