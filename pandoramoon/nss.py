"""nss: a compiled nested slice sampler for Pandora (experimental).

Description: NSS_SAMPLER.md; development history and validation: SAMPLING.md (Rounds 4-14) in the
project directory. `fit()` is the entry point (independent runs merged, self-checking); `run()` is one
nested-sampling run. Both take a numba-compiled function ll_u(u) of a point u in the unit cube
(prior transform inside). A builder of ll_u from a `pandoramoon.Likelihood` and user priors is not
part of the package yet (the scripts use scratch_r8/un_numba.make_ll_u).

Goal (SAMPLING.md, user 2026-10-08): sampler overhead -> 0, i.e. wall time
~ calls x kernel time. Everything in one numba function: no Python per
call and none per iteration. Python only builds the likelihood (as
un_numba.make_ll_u) and post-processes the dead points.

Algorithm (standard nested sampling; Skilling 2006; slice moves as in
PolyChord / UltraNest's slice samplers, Neal 2003):

- n live points in the unit cube; per iteration the k lowest are removed
  (k = 1: classic; k > 1: parallel nested sampling, Henderson & Goggans
  2014: shrinkage of the j-th removed point 1/(n - j)), and k new points
  are drawn above Lmin = the highest removed logL, each by a slice chain
  of `nsteps` accepted moves started at a random surviving live point.
  The k chains run in parallel threads (numba prange) when k > 1.
- Directions, dirs="pairs" (Round 7; fit()'s default since Round 8): v = +-(U_a - U_b) with a a random
  live point and b one of its `kpair` nearest neighbours (neighbour lists recomputed every `recluster` x n
  replacements, since Round 13 by pair_lists(): the same lists ~10x cheaper than the full K-nearest graph
  of cluster()). Local orientation and scale, but independent of the chain's position (exact).
- Directions, dirs="cluster" (run()'s default, Round 4): the live points are clustered every
  `recluster` x n replacements (connected components of the K-nearest-neighbour graph, K = 2 ndim,
  neighbours among <= nbsub random reference points; new points inherit their start point's cluster).
  Each slice step uses the difference of two random live points of the start point's cluster (start
  excluded). Clusters with < 2 other points fall back to the start point's K nearest neighbours.
- Directions, dirs="lcov" (Round 12): v = sqrt(2) L_a z with L_a the Cholesky factor of the covariance of
  a random live point's neighbour list (exact; no gain over "pairs").
- Optional extra steps: a fraction `paxis` along coordinate axes, `pglob` along global pair differences.
- Removed in Round 14 (superseded, user-approved cleanup): dirs="knn" (biased), "knnmh", "localcov" and the
  differential-evolution mode jumps `pjump`. They are kept in pandoramoon/nss_legacy.py (same API, frozen)
  and in git (tag nss-all-dirs).
- Slice (Neal 2003): an interval of width w = `scale` (in units of v)
  placed at random around the point, linear stepping-out by w per side
  (exact; doubling would need Neal's extra acceptance test), then
  shrinking. `scale` is adapted between iterations: x1.1 if there were
  more than ~1 expansion per step, /1.1 otherwise. Wrapped (periodic) dimensions are taken modulo 1.
- Evidence: logX after removing the j-th point of a batch decreases by
  1/(n - j); dead-point weights X_{i-1} - X_i; at the end the live points
  get X_final / n each. Stop when the live points' remaining evidence is
  < frac_remain of logZ (UltraNest default 0.01).

    from pandoramoon import nss
    import un_numba, problem                                  # project scripts (scratch_r8)
    P = problem.Problem("moon-phase"); ll = un_numba.make_ll_u(P)
    r = nss.fit(ll, P.ndim, wrapped=P.wrapped, nlive=1000, threads=16)
    r["passed"], r["logz"], r["logzerr"], r["samples_u"]      # merged equal-weight posterior (unit cube)
"""
import math
import time

import numpy as np
from numba import njit, prange

_SAMPLERS = {}
_HELPERS = {}


@njit(cache=True)
def _cholesky(C):
    """lower Cholesky factor of a symmetric positive definite matrix (plain loops: numba's np.linalg needs
    scipy, which is not a dependency of the package)"""
    n = C.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        s = C[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        L[j, j] = math.sqrt(max(s, 1e-300))
        for i in range(j + 1, n):
            t = C[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t / L[j, j]
    return L
@njit(cache=True)
def _eigh_jacobi(A, tol=1e-12, max_sweeps=60):
    """eigenvalues / eigenvectors (columns) of a symmetric matrix, cyclic Jacobi (no scipy / LAPACK needed)"""
    n = A.shape[0]
    a = A.copy()
    V = np.eye(n)
    for _sw in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p, q] * a[p, q]
        if off < tol * tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p, q]) < 1e-300:
                    continue
                th = 0.5 * (a[q, q] - a[p, p]) / a[p, q]
                t = (1.0 if th >= 0 else -1.0) / (abs(th) + math.sqrt(th * th + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                sn = t * c
                for k in range(n):
                    akp = a[k, p]
                    akq = a[k, q]
                    a[k, p] = c * akp - sn * akq
                    a[k, q] = sn * akp + c * akq
                for k in range(n):
                    apk = a[p, k]
                    aqk = a[q, k]
                    a[p, k] = c * apk - sn * aqk
                    a[q, k] = sn * apk + c * aqk
                for k in range(n):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - sn * vkq
                    V[k, q] = sn * vkp + c * vkq
    lam = np.empty(n)
    for i in range(n):
        lam[i] = a[i, i]
    return lam, V


MAXEXP = 100     # max linear step-outs per side and slice
INIT_VERSION = 2   # 2: rejection init above the LOGZERO plateau by default (Round 20); stored runs carry init_*
PLATEAU = -1e299  # logL at or below this is the LOGZERO plateau (outside the prior; sampling.LOGZERO = -1e300)


def make_sampler(ll_u, parallel, with_data=False):
    """compiled sampler for ll_u. with_data=False: ll_u(u); with_data=True: ll_u(u, D) with a data tuple D passed
    through run(data=D), so ONE compiled sampler serves many data sets (no recompilation per light curve)."""
    key = (id(ll_u), parallel, with_data)
    if key in _SAMPLERS:
        return _SAMPLERS[key]
    if with_data:
        LL = ll_u
    else:
        @njit(inline="always")
        def LL(x, D):
            return ll_u(x)

    @njit(inline="always")
    def wrap_inside(x, wrapped):
        for i in range(x.shape[0]):
            if wrapped[i]:
                x[i] -= math.floor(x[i])
            elif not (x[i] > 0.0 and x[i] < 1.0):
                return False
        return True

    @njit
    def k_smallest(vals, k, mask):
        """indices of the k smallest vals (ascending) among mask[i] == True
        (insertion into a sorted buffer: O(n k), no full sort)"""
        bd = np.full(k, np.inf)
        bi = np.full(k, -1, dtype=np.int64)
        m = 0
        for q in range(vals.shape[0]):
            if not mask[q]:
                continue
            d = vals[q]
            if m < k:
                p = m
                m += 1
            elif d < bd[k - 1]:
                p = k - 1
            else:
                continue
            while p > 0 and bd[p - 1] > d:
                bd[p] = bd[p - 1]
                bi[p] = bi[p - 1]
                p -= 1
            bd[p] = d
            bi[p] = q
        return bi[:m]

    @njit
    def neighbours(U, j, alive, sd, K, d2, M, idx):
        """U[j] and its K nearest alive points (metric scaled by sd), searched
        among all alive points if there are <= M, else among M random alive
        points (cost O(M ndim) per chain instead of O(n ndim))"""
        n = U.shape[0]
        if n <= M:
            m = 0
            for q in range(n):
                if alive[q] and q != j:
                    idx[m] = q
                    m += 1
        else:
            # M distinct points: start + i * stride (mod n) with gcd(stride, n) = 1
            while True:
                stride = 1 + np.random.randint(n - 1)
                a_, b_ = stride, n
                while b_:
                    a_, b_ = b_, a_ % b_
                if a_ == 1:
                    break
            q = np.random.randint(n)
            m = 0
            for _t in range(n):
                if alive[q] and q != j:
                    idx[m] = q
                    m += 1
                    if m == M:
                        break
                q = (q + stride) % n
        for a in range(m):
            q = idx[a]
            s = 0.0
            for i in range(U.shape[1]):
                t = (U[q, i] - U[j, i]) / sd[i]
                s += t * t
            d2[a] = s
        mask = np.ones(m, dtype=np.bool_)
        sel = k_smallest(d2[:m], K, mask)
        out = np.empty(K + 1, dtype=np.int64)
        out[0] = j
        for a in range(K):
            out[a + 1] = idx[sel[a]]
        return out

    @njit
    def k_smallest_all(vals, k):
        bd = np.full(k, np.inf)
        bi = np.full(k, -1, dtype=np.int64)
        m = 0
        for q in range(vals.shape[0]):
            d = vals[q]
            if m < k:
                p = m
                m += 1
            elif d < bd[k - 1]:
                p = k - 1
            else:
                continue
            while p > 0 and bd[p - 1] > d:
                bd[p] = bd[p - 1]
                bi[p] = bi[p - 1]
                p -= 1
            bd[p] = d
            bi[p] = q
        return bi[:m]

    @njit(parallel=parallel)
    def knn_graph(U, alive, sd, K, ref, nref, nbr):
        """nbr[q] = the K nearest points to U[q] among U[ref[:nref]] (q excluded)"""
        n, ndim = U.shape
        for q in prange(n):
            if not alive[q]:
                continue
            d2 = np.empty(nref)
            for a in range(nref):
                r = ref[a]
                if r == q:
                    d2[a] = np.inf
                    continue
                s = 0.0
                for i in range(ndim):
                    t = (U[r, i] - U[q, i]) / sd[i]
                    s += t * t
                d2[a] = s
            sel = k_smallest_all(d2, K)
            for t in range(K):
                nbr[q, t] = ref[sel[t]] if t < sel.shape[0] else q

    @njit
    def pair_lists(U, alive, sd, kk, M, nbr, ref, wmi):
        """dirs="pairs" (Round 13): nbr[q, :kk] = the kk nearest alive points to U[q] (metric scaled by sd) among
        the reference set (all alive points if n <= M, else M distinct random ones), ascending. The same lists as
        the first kk columns of cluster()'s K-nearest graph (checked: test_knnfast.py), without the union-find
        (the labels are not used by "pairs") and faster: scaled reference coordinates stored transposed, so the
        squared distances of one point to all references are summed dimension by dimension over contiguous
        arrays (SIMD), then one selection pass keeps the kk smallest; no allocation per point.
        wmi[i] True: minimal-image difference in dimension i (wrapped dimensions with minimg=True, Round 19 / E2)."""
        n, ndim = U.shape
        nref = refset(alive, -1, M, ref)
        inv = np.empty(ndim)
        for i in range(ndim):
            inv[i] = 1.0 / sd[i]
        VT = np.empty((ndim, nref))
        for a in range(nref):
            r = ref[a]
            for i in range(ndim):
                VT[i, a] = U[r, i] * inv[i]
        d = np.empty(nref)
        bd = np.empty(kk)
        bi = np.empty(kk, dtype=np.int64)
        for q in range(n):
            if not alive[q]:
                continue
            for a in range(nref):
                d[a] = 0.0
            for i in range(ndim):
                xi = U[q, i] * inv[i]
                row = VT[i]
                if wmi[i]:
                    for a in range(nref):
                        t = U[ref[a], i] - U[q, i]
                        t = (t - math.floor(t + 0.5)) * inv[i]
                        d[a] += t * t
                else:
                    for a in range(nref):
                        t = row[a] - xi
                        d[a] += t * t
            m = 0
            worst = np.inf
            for a in range(nref):
                s = d[a]
                if s >= worst or ref[a] == q:
                    continue
                if m < kk:
                    p = m
                    m += 1
                else:
                    p = kk - 1
                while p > 0 and bd[p - 1] > s:
                    bd[p] = bd[p - 1]
                    bi[p] = bi[p - 1]
                    p -= 1
                bd[p] = s
                bi[p] = ref[a]
                if m == kk:
                    worst = bd[kk - 1]
            for t in range(kk):
                nbr[q, t] = bi[t] if t < m else q

    @njit
    def find(parent, q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    @njit
    def cluster(U, alive, sd, K, M, labels, nbr, ref):
        """labels = connected components of the K-nearest-neighbour graph
        (neighbours among <= M distinct random alive reference points);
        returns the number of clusters"""
        n = U.shape[0]
        nref = 0
        if n <= M:
            for q in range(n):
                if alive[q]:
                    ref[nref] = q
                    nref += 1
        else:
            while True:
                stride = 1 + np.random.randint(n - 1)
                a_, b_ = stride, n
                while b_:
                    a_, b_ = b_, a_ % b_
                if a_ == 1:
                    break
            q = np.random.randint(n)
            for _t in range(n):
                if alive[q]:
                    ref[nref] = q
                    nref += 1
                    if nref == M:
                        break
                q = (q + stride) % n
        knn_graph(U, alive, sd, K, ref, nref, nbr)
        parent = np.arange(n)
        for q in range(n):
            if not alive[q]:
                continue
            for t in range(K):
                a = find(parent, q)
                b = find(parent, nbr[q, t])
                if a != b:
                    parent[a] = b
        root_label = np.full(n, -1, dtype=np.int64)
        nc = 0
        for q in range(n):
            if alive[q]:
                r = find(parent, q)
                if root_label[r] < 0:
                    root_label[r] = nc
                    nc += 1
                labels[q] = root_label[r]
        return nc

    @njit
    def refset(alive, j, M, R):
        """R[:nR] = all alive points (n <= M) or M distinct random alive ones"""
        n = alive.shape[0]
        nR = 0
        if n <= M:
            for q in range(n):
                if alive[q]:
                    R[nR] = q
                    nR += 1
            return nR
        while True:
            stride = 1 + np.random.randint(n - 1)
            a_, b_ = stride, n
            while b_:
                a_, b_ = b_, a_ % b_
            if a_ == 1:
                break
        q = np.random.randint(n)
        for _t in range(n):
            if alive[q]:
                R[nR] = q
                nR += 1
                if nR == M:
                    break
            q = (q + stride) % n
        return nR

    @njit
    def members(j, alive, labels, idx):
        """alive points of j's cluster except j -> idx[:m]; returns m"""
        m = 0
        lab = labels[j]
        for q in range(labels.shape[0]):
            if alive[q] and q != j and labels[q] == lab:
                idx[m] = q
                m += 1
        return m

    @njit(parallel=parallel)
    def local_chols(U, alive, nbr, wrapped, Lp):
        """Lp[q] = Cholesky factor of the covariance of q's neighbour list nbr[q] (minimal-image differences in
        wrapped dimensions; neighbours relative to q, mean removed), regularised"""
        n, ndim = U.shape
        K = nbr.shape[1]
        for q in prange(n):
            if not alive[q]:
                continue
            dd = np.empty((K, ndim))
            for t in range(K):
                r = nbr[q, t]
                for i in range(ndim):
                    d = U[r, i] - U[q, i]
                    if wrapped[i]:
                        d -= math.floor(d + 0.5)
                    dd[t, i] = d
            for i in range(ndim):
                mu = 0.0
                for t in range(K):
                    mu += dd[t, i]
                mu /= K
                for t in range(K):
                    dd[t, i] -= mu
            C = np.zeros((ndim, ndim))
            for i in range(ndim):
                for i2 in range(i + 1):
                    s_ = 0.0
                    for t in range(K):
                        s_ += dd[t, i] * dd[t, i2]
                    C[i, i2] = s_ / (K - 1)
                    C[i2, i] = C[i, i2]
            tr = 0.0
            for i in range(ndim):
                tr += C[i, i]
            for i in range(ndim):
                C[i, i] += 1e-10 * tr / ndim + 1e-300
            Lq = _cholesky(C)
            for i in range(ndim):
                for i2 in range(ndim):
                    Lp[q, i, i2] = Lq[i, i2]

    @njit
    def principal_axes(U, alive, wmi, PA):
        """PA[k] = sqrt(2 lam_k) e_k of the alive points' covariance (minimal image about the circular mean in the
        dimensions with wmi; plain otherwise)"""
        n, ndim = U.shape
        ref = np.zeros(ndim)
        for i in range(ndim):
            if wmi[i]:
                cs = 0.0
                sn = 0.0
                for a in range(n):
                    if alive[a]:
                        cs += math.cos(2.0 * math.pi * U[a, i])
                        sn += math.sin(2.0 * math.pi * U[a, i])
                ref[i] = math.atan2(sn, cs) / (2.0 * math.pi)
        m = 0
        mu = np.zeros(ndim)
        C = np.zeros((ndim, ndim))
        d = np.empty(ndim)
        for a in range(n):
            if not alive[a]:
                continue
            m += 1
            for i in range(ndim):
                t = U[a, i] - ref[i]
                if wmi[i]:
                    t -= math.floor(t + 0.5)
                d[i] = t
            for i in range(ndim):
                mu[i] += d[i]
                for i2 in range(i + 1):
                    C[i, i2] += d[i] * d[i2]
        for i in range(ndim):
            mu[i] /= m
        for i in range(ndim):
            for i2 in range(i + 1):
                C[i, i2] = C[i, i2] / m - mu[i] * mu[i2]
                C[i2, i] = C[i, i2]
        lam, V = _eigh_jacobi(C)
        for k in range(ndim):
            sc = math.sqrt(2.0 * max(lam[k], 1e-300))
            for i in range(ndim):
                PA[k, i] = sc * V[i, k]

    @njit
    def chain(U, j, L0, nb, Lmin, nsteps, scale, wrapped, x, xn, alive, dmode, paxis, sdv, nbr, kpair, pglob, D, Lp,
              ppca, PA):
        """slice chain at fixed Lmin from U[j] (logL L0); directions by dmode: 0 = differences of members of nb
        ("cluster"), 2 = local pairs ("pairs"), 6 = local pairs with the minimal image in wrapped dimensions
        (Round 19, SAMPLING_REVIEW.md E2), 3 = local covariance ("lcov"); optional axis / global-pair steps.
        Returns (L, ncalls, nexpand, nstuck, nmaxexp) with the end point in x; nstuck = steps whose shrink collapsed
        without an accepted point (the chain did not move), nmaxexp = step-outs that hit MAXEXP (B7)"""
        ndim = U.shape[1]
        n = U.shape[0]
        for i in range(ndim):
            x[i] = U[j, i]
        L = L0
        nc = 0
        nexp = 0
        nstuck = 0
        nmaxexp = 0
        m = nb.shape[0]
        v = np.empty(ndim)
        for _ in range(nsteps):
            a = np.random.randint(m)
            b = np.random.randint(m - 1)
            if b >= a:
                b += 1
            vv = 0.0
            ur = np.random.uniform(0.0, 1.0) if (paxis > 0.0 or pglob > 0.0 or ppca > 0.0) else 2.0
            if ur < paxis:
                # coordinate-axis slice (Gibbs-like), length ~ the live points' spread in that axis
                for i in range(ndim):
                    v[i] = 0.0
                ka = np.random.randint(ndim)
                v[ka] = 1.4142135623730951 * sdv[ka]
                vv = v[ka] * v[ka]
            elif ur < paxis + pglob + ppca and ur >= paxis + pglob:
                # principal axis of the live points' covariance (L5 / "de-mix", Buchner 2022): PA[k] = sqrt(2 lam_k) e_k,
                # computed at the last neighbour-list refresh from the frozen live set (independent of the chain: exact)
                ka = np.random.randint(ndim)
                for i in range(ndim):
                    v[i] = PA[ka, i]
                    vv += v[i] * v[i]
            elif ur < paxis + pglob:
                # global pair difference (two random live points): large moves along the global shape
                while True:
                    pa = np.random.randint(n)
                    if alive[pa]:
                        break
                while True:
                    pb = np.random.randint(n)
                    if alive[pb] and pb != pa:
                        break
                for i in range(ndim):
                    v[i] = U[pa, i] - U[pb, i]
                    vv += v[i] * v[i]
            elif dmode == 3:
                # local covariance of a random live point's neighbourhood (dirs="lcov"): v = sqrt(2) L_a z
                # (sqrt 2: the scale of a difference of two points); L_a from the last reclustering, inherited
                # by new points; independent of the chain's position (exact)
                while True:
                    pa = np.random.randint(n)
                    if alive[pa]:
                        break
                z = np.random.standard_normal(ndim)
                for i in range(ndim):
                    s_ = 0.0
                    for i2 in range(i + 1):
                        s_ += Lp[pa, i, i2] * z[i2]
                    v[i] = 1.4142135623730951 * s_
                    vv += v[i] * v[i]
            elif dmode == 2 or dmode == 6:
                # local pair difference: a random alive point and one of its kpair nearest neighbours
                # (neighbour lists from the last reclustering; independent of the chain's position)
                while True:
                    pa = np.random.randint(n)
                    if alive[pa]:
                        break
                pb = nbr[pa, np.random.randint(kpair)]
                sg = 1.0 if np.random.uniform(0.0, 1.0) < 0.5 else -1.0
                for i in range(ndim):
                    dv = U[pa, i] - U[pb, i]
                    if dmode == 6 and wrapped[i]:
                        dv -= math.floor(dv + 0.5)     # minimal image (still independent of the chain's position)
                    v[i] = sg * dv
                    vv += v[i] * v[i]
            else:
                for i in range(ndim):
                    v[i] = U[nb[a], i] - U[nb[b], i]
                    vv += v[i] * v[i]
            if vv == 0.0:          # duplicate points (should not happen): no direction
                continue
            left = -scale * np.random.uniform(0.0, 1.0)   # random placement of the initial interval
            right = left + scale
            # step out (linear)
            for _e in range(MAXEXP):
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * left
                if not wrap_inside(xn, wrapped):
                    break
                nc += 1
                if LL(xn, D) > Lmin:
                    left -= scale
                    nexp += 1
                    if _e == MAXEXP - 1:
                        nmaxexp += 1
                else:
                    break
            for _e in range(MAXEXP):
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * right
                if not wrap_inside(xn, wrapped):
                    break
                nc += 1
                if LL(xn, D) > Lmin:
                    right += scale
                    nexp += 1
                    if _e == MAXEXP - 1:
                        nmaxexp += 1
                else:
                    break
            # shrink
            while True:
                t = np.random.uniform(left, right)
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * t
                if wrap_inside(xn, wrapped):
                    nc += 1
                    Lx = LL(xn, D)
                    if Lx > Lmin:
                        for i in range(ndim):
                            x[i] = xn[i]
                        L = Lx
                        break
                if t < 0.0:
                    left = t
                else:
                    right = t
                if right - left < 1e-14:
                    nstuck += 1
                    break
        return L, nc, nexp, nstuck, nmaxexp

    @njit
    def grow(dU, dL, dlogw, nd):
        m = 2 * dU.shape[0]
        a = np.empty((m, dU.shape[1]))
        b = np.empty(m)
        c = np.empty(m)
        for q in range(nd):
            for i in range(dU.shape[1]):
                a[q, i] = dU[q, i]
            b[q] = dL[q]
            c[q] = dlogw[q]
        return a, b, c

    @njit(parallel=parallel, nogil=True)
    def ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed, scale0, M, dirmode, recluster, paxis, kpair,
           stop, pglob, D, logx0, scale_gain, freeze_after, minimg, ppca, exp_target):
        # dirmode: 1 = "cluster", 4 = "pairs", 5 = "lcov" (the numbers of the full version, nss_legacy)
        # Round 19 options (SAMPLING_REVIEW.md; defaults = the behaviour before): logx0 = ln X at the start (rejection
        # init, B1); scale_gain = factor of the scale adaptation per iteration (1.1; 1.0 = fixed scale), freeze_after =
        # iteration after which the scale is frozen (-1: never) (E1 / X3); minimg = minimal image in wrapped dimensions
        # for the pair metric, the pair directions and the metric's spread (circular sd) (E2 / X5)
        n, ndim = U.shape
        # seeds (per thread in parallel mode: numba seeds each thread's state)
        np.random.seed(seed)
        cap = 1 << 16
        dU = np.empty((cap, ndim))
        dL = np.empty(cap)
        dlogw = np.empty(cap)
        nd = 0
        logX = logx0
        logZ = -np.inf
        ncall = 0
        nstuck_tot = 0
        nmaxexp_tot = 0
        wmi = np.zeros(ndim, dtype=np.bool_)
        if minimg:
            for i in range(ndim):
                wmi[i] = wrapped[i]
        scale = scale0
        it = 0
        nexp_tot = 0
        nstep_tot = 0
        sd = np.empty(ndim)
        newU = np.empty((k, ndim))
        newL = np.empty(k)
        ncs = np.zeros(k, dtype=np.int64)
        nexps = np.zeros(k, dtype=np.int64)
        nstucks = np.zeros(k, dtype=np.int64)
        nmaxexps = np.zeros(k, dtype=np.int64)
        starts = np.empty(k, dtype=np.int64)
        xs = np.empty((k, ndim))
        xns = np.empty((k, ndim))
        alive = np.ones(n, dtype=np.bool_)
        d2s = np.empty((k, n))
        idxs = np.empty((k, n), dtype=np.int64)
        sd_every = max(1, n // (10 * k))
        labels = np.zeros(n, dtype=np.int64)
        nbr = np.empty((n, K), dtype=np.int64)
        Lp = np.zeros((n if dirmode == 5 else 1, ndim, ndim))
        PA = np.zeros((ndim, ndim))
        for i in range(ndim):
            PA[i, i] = 0.1
        for q0 in range(n):
            for t0 in range(K):
                nbr[q0, t0] = q0
        ref = np.empty(n, dtype=np.int64)
        mcount = np.zeros(k, dtype=np.int64)
        nclus = 1
        nclus_max = 1
        nrecl = 0
        repl_since = n * 10           # force clustering at the first iteration
        nfallback = 0
        ranks = np.zeros(k, dtype=np.int64)
        mix = np.zeros(ndim)
        nmix = 0
        ins = np.empty(1 << 16, dtype=np.int64)
        nins = 0
        while it < max_iter:
            if stop[0] != 0:                       # abort requested by fit() (the run is discarded)
                break
            sel = k_smallest(L, k, alive)          # ascending
            Lmin = L[sel[k - 1]]
            # record the k removed points
            for jj in range(k):
                q = sel[jj]
                logX_new = logX - 1.0 / (n - jj)
                logw = logX + math.log(-math.expm1(-1.0 / (n - jj)))    # X_old - X_new
                if nd >= dU.shape[0]:
                    dU, dL, dlogw = grow(dU, dL, dlogw, nd)
                dU[nd] = U[q]
                dL[nd] = L[q]
                dlogw[nd] = logw
                nd += 1
                logZ = np.logaddexp(logZ, logw + L[q])
                logX = logX_new
            # stopping: remaining live evidence vs accumulated
            Lmax = L.max()
            if np.isfinite(logZ) and logX + Lmax - logZ < math.log(frac_remain):
                break
            # spread of the live points (for the neighbour metric), every n/10 replacements
            if it % sd_every == 0:
                for i in range(ndim):
                    if wmi[i]:
                        # circular spread: sqrt(-2 ln R) / (2 pi), R = |mean exp(2 pi i u)| (a seam-split clump is one clump)
                        cs = 0.0
                        sn = 0.0
                        for a in range(n):
                            cs += math.cos(2.0 * math.pi * U[a, i])
                            sn += math.sin(2.0 * math.pi * U[a, i])
                        R = math.sqrt(cs * cs + sn * sn) / n
                        sd[i] = min(math.sqrt(-2.0 * math.log(max(R, 1e-300))) / (2.0 * math.pi),
                                    0.28867513459481287) + 1e-300
                        continue
                    mu = 0.0
                    for a in range(n):
                        mu += U[a, i]
                    mu /= n
                    s = 0.0
                    for a in range(n):
                        s += (U[a, i] - mu) ** 2
                    sd[i] = math.sqrt(s / n) + 1e-300
            for jj in range(k):
                alive[sel[jj]] = False
            if dirmode == 4 and repl_since >= recluster * n:
                pair_lists(U, alive, sd, kpair, M, nbr, ref, wmi)
                if ppca > 0.0:
                    principal_axes(U, alive, wmi, PA)
                nrecl += 1
                repl_since = 0
            if (dirmode == 1 or dirmode == 5) and repl_since >= recluster * n:
                nclus = cluster(U, alive, sd, K, M, labels, nbr, ref)
                if dirmode == 5:
                    local_chols(U, alive, nbr, wrapped, Lp)
                nclus_max = max(nclus_max, nclus)
                nrecl += 1
                repl_since = 0
            repl_since += k
            if k > 1:
                # reproducible parallel runs (Round 17): numba keeps one random state per thread, and which thread
                # runs which chain is not fixed. Each chain therefore reseeds its thread's state from (seed, iteration,
                # chain), and the main thread is reseeded after the parallel loop (below).
                np.random.seed((seed * 1000003 + 2 * it + 1) % 4294967296)
            for c in range(k):
                while True:
                    j = np.random.randint(n)
                    if alive[j]:
                        break
                starts[c] = j
            for c in prange(k):
                if k > 1:
                    np.random.seed((seed * 2654435761 + it * k + c) % 4294967296)
                m = 0
                if dirmode == 1:
                    m = members(starts[c], alive, labels, idxs[c])
                if dirmode == 4 or dirmode == 5:
                    mem = idxs[c, :2]
                    mcount[c] = 0
                elif m >= 2:
                    mem = idxs[c, :m]
                    mcount[c] = 0
                else:
                    mem = neighbours(U, starts[c], alive, sd, K, d2s[c], M, idxs[c])
                    mcount[c] = 1 if dirmode == 1 else 0
                Lc, nc, ne, nst, nmx = chain(U, starts[c], L[starts[c]], mem, Lmin, nsteps, scale, wrapped, xs[c], xns[c],
                                             alive, (6 if minimg else 2) if dirmode == 4 else (3 if dirmode == 5 else 0),
                                             paxis, sd, nbr, kpair, pglob, D, Lp, ppca, PA)
                newL[c] = Lc
                ncs[c] = nc
                nexps[c] = ne
                nstucks[c] = nst
                nmaxexps[c] = nmx
                for i in range(ndim):
                    newU[c, i] = xs[c, i]
                # insertion index (Fowlie, Handley & Su 2020): rank of the new logL among the n - k
                # surviving live points; uniform on {0..n-k} if the new point is a fair draw above Lmin
                r = 0
                for q2 in range(n):
                    if alive[q2] and L[q2] < Lc:
                        r += 1
                ranks[c] = r
            if nins + k > ins.shape[0]:
                ins2 = np.empty(2 * ins.shape[0] + k, dtype=np.int64)
                for q2 in range(nins):
                    ins2[q2] = ins[q2]
                ins = ins2
            for c in range(k):
                ins[nins] = ranks[c]
                nins += 1
            for c in range(k):                     # mixing diagnostic: (end - start)^2 / (2 var_live) per dim
                for i in range(ndim):
                    dlt = newU[c, i] - U[starts[c], i]
                    if wrapped[i]:
                        dlt -= math.floor(dlt + 0.5)
                    mix[i] += dlt * dlt / (2.0 * sd[i] * sd[i])
                nmix += 1
            for c in range(k):
                q = sel[c]
                for i in range(ndim):
                    U[q, i] = newU[c, i]
                L[q] = newL[c]
                labels[q] = labels[starts[c]]
                for t0 in range(K):                # new point inherits its start's neighbour list
                    nbr[q, t0] = nbr[starts[c], t0]
                if dirmode == 5:                   # ... and its local covariance factor
                    for i in range(ndim):
                        for i2 in range(ndim):
                            Lp[q, i, i2] = Lp[starts[c], i, i2]
                alive[q] = True
                nfallback += mcount[c]
                ncall += ncs[c]
                nexp_tot += nexps[c]
                nstuck_tot += nstucks[c]
                nmaxexp_tot += nmaxexps[c]
            nstep_tot += k * nsteps
            # scale adaptation: aim at ~1 expansion per step (interval comparable to the slice)
            if freeze_after < 0 or it < freeze_after:
                if nexps.sum() > exp_target * k * nsteps:
                    scale *= scale_gain
                else:
                    scale /= scale_gain
                scale = min(max(scale, 1e-3), 1e3)
            it += 1
        # live points
        logw_live = logX - math.log(n)
        for a in range(n):
            if nd >= dU.shape[0]:
                dU, dL, dlogw = grow(dU, dL, dlogw, nd)
            dU[nd] = U[a]
            dL[nd] = L[a]
            dlogw[nd] = logw_live
            nd += 1
        return (dU[:nd], dL[:nd], dlogw[:nd], ncall, it, scale, nexp_tot / max(nstep_tot, 1),
                nclus, nclus_max, nrecl, nfallback, ins[:nins], mix / max(nmix, 1), nstuck_tot, nmaxexp_tot)

    @njit(parallel=parallel, nogil=True)
    def init(n, ndim, seed, D):
        np.random.seed(seed)
        U = np.random.uniform(0.0, 1.0, (n, ndim))
        L = np.empty(n)
        for a in prange(n):
            L[a] = LL(U[a], D)
        return U, L

    _SAMPLERS[key] = (ns, init)
    _HELPERS[key] = dict(pair_lists=pair_lists, cluster=cluster)      # for tests (test_knnfast.py)
    return ns, init


def _ks_p(D, N):
    """asymptotic Kolmogorov p-value (Stephens 1970 correction)"""
    if N <= 0:
        return 1.0
    lam = (math.sqrt(N) + 0.12 + 0.11 / math.sqrt(N)) * D
    p = 0.0
    for j in range(1, 101):
        p += 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
    return float(min(max(p, 0.0), 1.0))


def insertion_test(ranks, m, nlive, seed=0):
    """Insertion-index cross-check (Fowlie, Handley & Su 2020, MNRAS 497, 5256): the rank of
    each new point among the m = n - k surviving live points is uniform on {0..m}. Returns the KS
    distance and p-value of all ranks (jittered to a continuous uniform), and per chunk of
    nlive insertions: the minimum p and the fraction of chunks with p < 0.01 (expected 0.01)."""
    if len(ranks) == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed + 7)
    u = (ranks + rng.uniform(size=len(ranks))) / (m + 1)

    def ks(x):
        x = np.sort(x)
        n = len(x)
        i = np.arange(1, n + 1)
        return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    D = ks(u)
    ps = []
    for a in range(0, len(u) - nlive + 1, nlive):
        ps.append(_ks_p(ks(u[a:a + nlive]), nlive))
    ps = np.array(ps) if ps else np.array([1.0])
    return dict(n=int(len(u)), D=D, p=_ks_p(D, len(u)), chunk=int(nlive), nchunks=int(len(ps)),
                chunk_p_min=float(ps.min()), chunk_frac_p01=float(np.mean(ps < 0.01)),
                mean=float(u.mean()))


def run(ll_u, ndim, wrapped=None, nlive=1000, nsteps=None, K=None, k=1, frac_remain=0.01,
        max_iter=10_000_000, seed=1, scale0=1.0, threads=None, nbsub=2000, dirs="cluster",
        recluster=0.25, paxis=0.0, kpair=4, stop=None, pglob=0.0, data=None, init_reject=True,
        init_min_draws=100_000, logx0=None, scale_gain=1.1, freeze_after=-1, minimg=False, ppca=0.0, exp_target=1.0):
    """returns dict(logz, logzerr, ncall, niter, wall, samples_u (equal weight), dead_u, dead_L, logw).
    Round 19 options (SAMPLING_REVIEW.md):
    init_reject (default True since Round 20; False = the initial live set of Rounds 1-19): draw the initial live points only where logL > PLATEAU (the LOGZERO plateau, B1) and start at
      ln X0 = ln(accepted / drawn) over >= init_min_draws draws (or the given logx0, e.g. an analytic 1 - f);
      the draws are counted in ncall. Without it, plateau points enter the live set (n_plateau in the result).
    scale_gain, freeze_after: scale adaptation per iteration (x/: scale_gain; 1.0 = fixed scale0) and the iteration
      after which it is frozen (-1: never) (E1, experiment X3).
    minimg: minimal image in wrapped dimensions for the pair metric, the pair directions and the spread (E2, X5).
    ppca (Round 21, NEW_LEADS L5, "de-mix"): fraction of slice steps along a random principal axis of the live points'
      covariance (sqrt(2 lam) e_k), recomputed at every neighbour-list refresh (dirs="pairs"); 0 = off.
    exp_target (Round 21, NEW_LEADS L6): step-outs per slice step that the scale adaptation aims at (1.0 = Rounds 1-20;
      Yallup et al. 2026: the expected calls per step are minimal at ~0.74 step-outs, i.e. an interval ~1.36 chords).
    data: None -> ll_u(u); a tuple D -> ll_u(u, D) (one compiled sampler for many data sets).
    dirs: "pairs" (fit's default), "cluster", "lcov". The superseded "knn", "knnmh", "localcov" and the mode jumps
    `pjump` were removed in Round 14; they are in pandoramoon.nss_legacy (same API) and git tag nss-all-dirs."""
    import numba
    parallel = k > 1
    assert 1 <= k <= nlive // 4, "k (points replaced per iteration) must be <= nlive / 4"
    if threads is not None:
        numba.set_num_threads(threads)
    ns, init = make_sampler(ll_u, parallel, data is not None)
    D = data if data is not None else (0,)
    wrapped = np.array(wrapped if wrapped is not None else [False] * ndim, dtype=np.bool_)
    nsteps = nsteps or 4 * ndim
    K = K or 2 * ndim
    t0 = time.perf_counter()
    ninit = nlive
    if init_reject:
        Us, Ls, nacc, ndraw, j = [], [], 0, 0, 0
        while nacc < nlive or ndraw < init_min_draws:
            m = max(nlive, min(init_min_draws, 100_000))
            U_, L_ = init(m, ndim, (seed * 7919 + 1_000_003 * (j + 1)) % 4294967296, D)
            ok = L_ > PLATEAU
            nacc += int(ok.sum())
            ndraw += m
            if sum(len(q) for q in Ls) < nlive:
                Us.append(U_[ok])
                Ls.append(L_[ok])
            j += 1
            if j > 10_000:
                raise RuntimeError("init_reject: almost no prior volume above the LOGZERO plateau")
        U = np.ascontiguousarray(np.concatenate(Us)[:nlive])
        L = np.ascontiguousarray(np.concatenate(Ls)[:nlive])
        lx0 = float(logx0) if logx0 is not None else math.log(nacc / ndraw)
        ninit = ndraw
    else:
        U, L = init(nlive, ndim, seed, D)
        nacc, ndraw = 0, 0
        lx0 = float(logx0) if logx0 is not None else 0.0
    n_plateau = int(np.sum(L <= PLATEAU))
    t1 = time.perf_counter()
    if dirs not in ("cluster", "pairs", "lcov"):
        raise ValueError(f"dirs={dirs!r}: removed in Round 14 (superseded); use pandoramoon.nss_legacy")
    (dU, dL, dlogw, ncall, it, scale, exp_rate, nclus, nclus_max, nrecl,
     nfallback, ins, mix, nstuck, nmaxexp) = ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed + 1, scale0,
                                              nbsub, {"cluster": 1, "pairs": 4, "lcov": 5}[dirs], recluster, paxis, kpair,
                                              stop if stop is not None else np.zeros(1, dtype=np.int64), pglob, D,
                                              lx0, float(scale_gain), int(freeze_after), bool(minimg), float(ppca),
                                              float(exp_target))
    t2 = time.perf_counter()
    lw = dlogw + dL
    logz = float(np.logaddexp.reduce(lw))
    w = np.exp(lw - logz)
    # information and error (Skilling): sqrt(H / n)
    fin = w > 0
    H = float(np.sum(w[fin] * (dL[fin] - logz)))
    # equal-weight posterior (systematic resampling)
    rng = np.random.default_rng(seed)
    cw = np.cumsum(w)
    cw /= cw[-1]
    nsamp = int(np.exp(-np.sum(w[fin] * np.log(w[fin]))))   # ~ESS-ish (Kish would be 1/sum w^2)
    ess = float(1.0 / np.sum(w ** 2))
    pos = (rng.uniform() + np.arange(nsamp)) / nsamp
    samples = dU[np.searchsorted(cw, pos)]
    insd = insertion_test(ins, nlive - k, nlive, seed)
    return dict(seed=int(seed), nlive=int(nlive), ins_u=((ins + np.random.default_rng(seed + 11).uniform(size=len(ins))) / (nlive - k + 1)
                                       ).astype(np.float32), mix=mix, insertion=insd, logz=logz, logzerr=math.sqrt(max(H, 0.0) / nlive), H=H, ncall=int(ncall) + ninit, niter=int(it),
                wall=t2 - t0, wall_init=t1 - t0, samples_u=samples, ess=ess, dead_u=dU, dead_L=dL, logw=dlogw,
                maxL=float(dL.max()), final_scale=scale, expand_rate=exp_rate,
                nclusters=int(nclus), nclusters_max=int(nclus_max), nrecluster=int(nrecl),
                nfallback=int(nfallback), dirs=dirs, logx0=lx0, init_acc=int(nacc), init_draws=int(ndraw),
                n_plateau=n_plateau, nstuck=int(nstuck), nmaxexp=int(nmaxexp))


# ---------------------------------------------------------------------------------------------
# fit(): M independent runs in parallel (one per thread, each serial), merged, with self-checks
# (SAMPLING.md Round 6: merged runs ~ one run with more live points; reproducibility is checked
# between runs, not inside one).

def _ks(a, b):
    a = np.sort(a)
    b = np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a)
                               - np.searchsorted(b, x, side="right") / len(b))))


def merge_weights(rs):
    """Merge independent runs into ONE nested-sampling run (Skilling 2006 Sect. 9; Higson et al. 2018,
    dynesty merge_runs): all points sorted by logL; the number of live points just before point j is
    N_live = sum n_i minus the final live points (of any run) already passed; the shrinkage per point
    is exp(-1/N_live) as in run(). The live count only drops at final live points (a dead point of a
    run is replaced by a new one of the same run). Returns (logZ, ln w + ln L per point, order)."""
    Ls, fl = [], []
    for r in rs:
        n = r.get("nlive")
        L = r["dead_L"]
        f = np.zeros(len(L), dtype=bool)
        f[len(L) - n:] = True                   # run() appends its n final live points last
        Ls.append(L)
        fl.append(f)
    L = np.concatenate(Ls)
    f = np.concatenate(fl)
    o = np.argsort(L, kind="stable")
    Lo = L[o]
    fo = f[o]
    ntot = int(f.sum())
    nl = ntot - (np.cumsum(fo) - fo)            # live points just before removing point j
    dlx = 1.0 / nl
    # start volume (Round 19, init_reject): pooled acceptance fraction of the runs' rejection draws, else the mean logx0
    if all(r.get("init_draws", 0) > 0 for r in rs):
        lx0 = math.log(sum(r["init_acc"] for r in rs) / sum(r["init_draws"] for r in rs))
    else:
        lx0 = float(np.mean([r.get("logx0", 0.0) for r in rs]))
    lx = lx0 - np.cumsum(dlx)
    lxprev = lx + dlx
    lw = lxprev + np.log(-np.expm1(-dlx))       # X_{j-1} - X_j
    lwl = np.empty(len(L))
    lwl[o] = lw + Lo
    return float(np.logaddexp.reduce(lwl)), lwl


def merge(rs, nsamp=None, seed=0, method="run"):
    """Merge of independent runs (SAMPLING.md Round 8).
    method="run" (default): as ONE run with sum n_i live points (merge_weights); logZ ~ mean of the runs'
    lnZ, which is centred on the truth in the known-Z tests (test_fit_known.py). Not robust to runs that
    never found the best mode: fit() excludes those first (failed_runs).
    method="zmean" (Round 6b/7): each run's normalised posterior x Z_i / sum Z, logZ = log mean Z_i;
    biased high by ~ var(lnZ_i) / 2 (Jensen; +0.1..+0.4 nats in the tests, +1.2 on C0 at nsteps 16).
    Returns (logZ, equal-weight samples_u, ESS)."""
    if method == "run":
        lzm, lw = merge_weights(rs)
    else:
        lz = np.array([r["logz"] for r in rs])
        lzm = np.logaddexp.reduce(lz) - math.log(len(rs))
        lw = np.concatenate([r["logw"] + r["dead_L"] - r["logz"] + (r["logz"] - lzm) - math.log(len(rs)) for r in rs])
    w = np.exp(lw - np.logaddexp.reduce(lw))
    ess = float(1.0 / np.sum(w ** 2))
    n = nsamp or int(min(ess, 200_000))
    rng = np.random.default_rng(seed)
    cw = np.cumsum(w)
    cw /= cw[-1]
    pos = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(cw, pos)
    # gather the selected points from the runs without stacking all of them (a full vstack copied
    # ~0.4 GB per merge at ~50 runs; fit() merges 5x per check)
    off = np.cumsum([0] + [len(r["dead_L"]) for r in rs])
    ir = np.searchsorted(off, idx, side="right") - 1
    out = np.empty((n, rs[0]["dead_u"].shape[1]))
    for q in np.unique(ir):
        m = ir == q
        out[m] = rs[q]["dead_u"][idx[m] - off[q]]
    return float(lzm), out, ess


def split_half_ks(rs, sub=40_000, seed=0, post_map=None):
    """max over parameters of the KS distance between the merged posteriors of the even- and
    odd-numbered runs (reproducibility of a merge of M/2 runs; the full merge is better by ~sqrt 2).
    post_map: unit-cube samples -> the (physical) coordinates in which to compare, e.g. folding
    exact symmetry copies whose relative weight is meaningless."""
    if len(rs) < 2:
        return np.nan, -1
    _, A, _ = merge(rs[0::2], nsamp=sub, seed=seed)
    _, B, _ = merge(rs[1::2], nsamp=sub, seed=seed + 1)
    if post_map is not None:
        A, B = post_map(A), post_map(B)
    d = [_ks(A[:, i], B[:, i]) for i in range(A.shape[1])]
    j = int(np.argmax(d))
    return float(d[j]), j


def failed_runs(rs, dl_max=5.0, dz_min=2.0):
    """indices of runs that never found the best mode: maxL more than dl_max below the best run's AND lnZ
    more than max(5 MAD-sd, dz_min) below the median (C0 at 150 live: 2 of 10 runs at lnZ -204 vs -186)"""
    if len(rs) < 3:
        return []
    z = np.array([r["logz"] for r in rs])
    mL = np.array([r["maxL"] for r in rs])
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med))
    return [i for i in range(len(rs)) if mL[i] < mL.max() - dl_max and z[i] < med - max(5 * mad, dz_min)]


def _lnz_stats(rs):
    """per-arm statistics over the good runs: mean / sd / MAD-sd of lnZ, err = sd / sqrt(M); also the
    log-mean-Z (for comparison) and the number of failed runs"""
    bad = set(failed_runs(rs))
    z = np.array([r["logz"] for i, r in enumerate(rs) if i not in bad])
    M = len(z)
    sd = float(z.std(ddof=1)) if M > 1 else np.nan
    mad = float(1.4826 * np.median(np.abs(z - np.median(z)))) if M > 1 else np.nan
    return dict(M=M, mean=float(z.mean()), sd=sd, err=sd / math.sqrt(M) if M > 1 else np.inf, mad_sd=mad,
                zmean=float(np.logaddexp.reduce(z) - math.log(M)), n_failed=len(bad), M_all=len(rs),
                good=[r for i, r in enumerate(rs) if i not in bad])


def _ks_sets(ra, rb, post_map, sub=40_000, seed=0):
    _, A, _ = merge(ra, nsamp=sub, seed=seed)
    _, B, _ = merge(rb, nsamp=sub, seed=seed + 1)
    if post_map is not None:
        A, B = post_map(A), post_map(B)
    d = [_ks(A[:, i], B[:, i]) for i in range(A.shape[1])]
    j = int(np.argmax(d))
    return float(d[j]), j


def ks_null(ra, rb, post_map, nperm=19, sub=20_000, seed=0):
    """noise distribution of the KS distance between the merged arms ra and rb (Round 16): KS distances between
    random re-splits of the pooled runs into two sets of the arms' sizes (if both arms sample the same posterior,
    their runs are exchangeable). Returns an array of nperm values."""
    rng = np.random.default_rng(seed)
    allr = list(ra) + list(rb)
    na = len(ra)
    out = []
    for k in range(nperm):
        idx = rng.permutation(len(allr))
        A = [allr[i] for i in sorted(idx[:na])]
        B = [allr[i] for i in sorted(idx[na:])]
        out.append(_ks_sets(A, B, post_map, sub=sub, seed=seed + 31 * k)[0])
    return np.array(out)


def pooled_insertion(rs):
    """insertion-index test pooled over runs: KS of all jittered insertion ranks (uniform for exact NS) and
    the Fisher combination of the per-run p-values. Returns (D, p_pooled, p_fisher)."""
    u = np.concatenate([r["ins_u"] for r in rs if "ins_u" in r]).astype(np.float64)
    if len(u) == 0:
        return np.nan, np.nan, np.nan
    u.sort()
    n = len(u)
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))
    ps = np.array([max(r["insertion"]["p"], 1e-300) for r in rs])
    chi2 = -2.0 * np.sum(np.log(ps))
    # chi2 with 2M dof: Wilson-Hilferty normal approximation of the upper tail
    k = 2 * len(ps)
    z = ((chi2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    p_f = float(0.5 * math.erfc(z / math.sqrt(2)))
    return D, _ks_p(D, n), p_f


def bootstrap_logz(rs, B=40, seed=0):
    """sd of the merged logZ over bootstrap resamples of the runs (Higson et al. 2018)"""
    rng = np.random.default_rng(seed)
    v = [merge_weights([rs[q] for q in rng.integers(0, len(rs), len(rs))])[0] for _ in range(B)]
    return float(np.std(v, ddof=1))


# accuracy presets of fit() (SAMPLING.md Round 12)
ACCURACY = {"high": dict(nlive=1000, lz_tol=0.1, ks_tol=0.05, min_runs=16),
            "medium": dict(nlive=500, lz_tol=0.2, ks_tol=0.08, min_runs=8)}


def fit(ll_u, ndim, wrapped=None, nlive=2000, nsteps=None, threads=12, batch=None, min_runs=None,
        max_runs=96, ks_tol=0.05, lz_tol=0.1, seed=1, dirs="pairs", frac_remain=0.01, verbose=True, log=None,
        post_map=None, paxis=0.0, kpair=4, check=True, max_nsteps=None, pglob=0.0, recluster=0.25, data=None,
        accuracy=None, resume=None, callback=None, check_rule="r16", run_kw=None, max_nlive=None):
    """Nested sampling with built-in convergence checks (SAMPLING.md Rounds 6b, 8).

    n_live escalation (Round 20, SAMPLING_REVIEW.md B3 / X4): runs that miss a compact mode bias lnZ low by up to the
    mode's mass; longer chains do not help, more live points do. If an arm with >= max(12, min_runs) runs shows a
    mode problem -- a heavy-tailed per-run lnZ distribution (sd > 3 x MAD-sd: a minority of runs found a better
    mode; >= max(12, min_runs) runs) or a large scatter (sd > 2.75 x the quoted sqrt(H/n); >= max(16, 2 min_runs) runs;
    normal at nsteps 64: 1.1-2.1) that does not fall with the chain length -- n_live is
    doubled at the same nsteps and both arms restart (up to max_nlive, default 4 nlive); the nsteps escalation is
    blocked in that evaluation (its dlnZ / KS tests compare which arm got the lucky runs; Round 18: a Kepler no-moon
    fit escalated 4x to nsteps 1024 that way). At max_nlive the fit continues without escalating
    (result "multimodal": True). Offline on run pools (subsets of 32 runs): false triggers 0 % on C0, C7, K1, HH22,
    KZ, KZR; detection 71 % (two modes, volume 1e-2, equal mass, 1000 live), 100 % at 4000 live. If the per-run
    variance falls like 1/n_live, a doubled n_live needs half the runs, so a false trigger costs only the discarded runs.
    Not detectable: a mode that every run misses (the lnZ scatter is then normal).

    accuracy: None (use the arguments) or a preset of ACCURACY (overrides nlive, lz_tol, ks_tol, min_runs):
    "high" (logZ +-0.1 statistical, KS 0.05, 1000 live), "medium" (+-0.2, KS 0.08, 500 live, >= 8 runs).

    Independent serial nss runs (n_live = nlive each) run in parallel threads; they are merged as one
    run (merge_weights). Two arms: the result arm at `nsteps` (default 4 ndim) and, if check=True, a
    check arm at nsteps / 2. Batches are added until, for the result arm,
      - the empirical logZ error sd(lnZ_i) / sqrt(M) <= lz_tol,
      - the split-half KS of the merged posterior <= ks_tol (max over parameters, post_map coords),
    and, for the nsteps check, the arms agree: |lnZ(N/2) - lnZ(N)| <= max(lz_tol, 2 sigma) and
    KS(arm N/2, arm N) <= ks_tol. If the arms disagree significantly (|dlnZ| > lz_tol and > 3 sigma,
    or KS between the arms > max(ks_tol, 2 x the split-half KS); 1.5x triggered on noise at 6 runs, 2 ks_tol left
    a grey zone in which C0 ran to max_runs), the chains are too short: nsteps is
    doubled, the old result arm becomes the check arm (no runs are lost except the old check arm), and
    sampling continues. With a bias that falls at least like 1/nsteps, dlnZ(N/2, N) bounds the
    remaining bias at N. Stops at max_runs (result arm) with passed=False if the checks fail.
    Returns dict(logz (merged result arm), logzerr (empirical), logzerr_boot, samples_u, runs, history,
    passed, nsteps, dlnz_check, ...).

    check_rule: "r16" (default since Round 16) or "r8" (the rule of Rounds 8-15). "r16" judges the SIZE of the
    difference between the arms against the tolerances, not its significance (SAMPLING.md Round 16): the KS
    distance between the arms is compared with ks_tol after removing its noise (permutation null of the arms'
    KS, ks_null): pass the KS part if ksa^2 - median(null)^2 <= ks_tol^2, escalate if ksa^2 - max(null)^2 >
    ks_tol^2, otherwise add runs. dlnZ: pass if |d| + sigma <= 2 lz_tol, escalate if |d| - 2 sigma > 2 lz_tol
    (3 sigma while an arm has fewer than min_runs runs: the sd of lnZ from few runs is unreliable), grey zone
    (own criteria met): escalate if sigma <= 0.6 lz_tol and |d| - sigma > lz_tol, or if both arms have >= 2 min_runs
    runs and the noise-corrected KS (median null) still exceeds ks_tol; and escalate if the per-run lnZ sd exceeds
    3.5 x the quoted sqrt(H/n) (>= max(16, 2 min_runs) runs): the chains do not mix. Offline simulations on pools of real
    runs: false escalations C0 25 % -> 2 %, C7 47 % -> 0 % (cost -13 %), logZ unchanged.

    Resume (Round 15, used by pandoramoon.sampling): callback(event, **kw) is called in the main thread with
    event "run" (r=run dict, nsteps=arm, idx=run index) for every finished run that enters an arm, and "state"
    (N=current result nsteps, nesc, dropped=nsteps of a dropped arm, dropped_idx=the run indices of that arm and of
    the runs aborted with it) after every escalation. resume = dict(..., skip=[indices not to reuse]) +
    dict(runs=[(nsteps, idx, run dict), ...], N=..., nesc=...) restarts from stored runs: the runs of the arms N
    and N/2 are reloaded, new runs take the smallest unused indices (run seed = seed + 1000 idx; the runs lost in
    flight at a crash are redone with their original seeds), and
    the checks are evaluated before any new run is started (a fit that had passed passes again at once)."""
    from concurrent.futures import ThreadPoolExecutor
    if accuracy is not None:
        a = ACCURACY[accuracy]
        nlive, lz_tol, ks_tol = a["nlive"], a["lz_tol"], a["ks_tol"]
        min_runs = min_runs or a["min_runs"]
    nsteps = nsteps or 4 * ndim
    max_nsteps = max_nsteps or 64 * ndim
    min_runs = min_runs or max(threads, 4)
    nlive0 = nlive
    max_nlive = max_nlive or 4 * nlive
    nlesc = 0
    multimodal = [False]
    tc = time.perf_counter()
    make_sampler(ll_u, False, data is not None)  # compile once, outside the threads
    run(ll_u, ndim, wrapped=wrapped, nlive=max(40, 2 * ndim + 8), nsteps=2, k=1, max_iter=3, seed=seed + 999, data=data,
        dirs=dirs, kpair=min(kpair, 4), init_reject=False)
    t_compile = time.perf_counter() - tc         # numba compile (first call per process); not in `wall`
    arms = {}                                    # nsteps -> list of runs
    hist = []
    t0 = time.perf_counter()
    nxt = 0
    N = nsteps
    nesc = 0
    ncall_tot = [0]
    used = set()                                 # run indices consumed (seed = seed + 1000 idx)
    if resume is not None:
        N = int(resume.get("N", nsteps))
        nesc = int(resume.get("nesc", 0))
        nlive = int(resume.get("nlive") or nlive)
        nlesc = int(resume.get("nlesc", 0))
        # indices of runs of dropped arms (and runs aborted with them) are not reused; indices of runs that were in
        # flight at a crash are (the same seeds as in the interrupted fit, Round 17)
        used |= set(int(i) for i in resume.get("skip", []))
        for ns_, idx_, r_ in resume.get("runs", []):
            used.add(int(idx_))
            r_["idx"] = int(idx_)
            if int(r_.get("nlive", nlive)) != nlive:
                continue
            if ns_ == N or (check and ns_ == max(1, N // 2)):
                arms.setdefault(ns_, []).append(r_)
        for v_ in arms.values():
            v_.sort(key=lambda q: q["seed"])

    def precheck(N):
        """cheap necessary condition (scalars only: lnZ statistics of both arms) for evaluate() to pass or to
        escalate by the dlnZ rule; the KS-based escalation is covered by the regular evaluation every `threads`
        finished runs (Round 13)"""
        Nc = max(1, N // 2)
        R = arms.get(N, [])
        if len(R) < 2:
            return False
        sR = _lnz_stats(R)
        if sR["M"] >= min_runs and sR["err"] <= lz_tol:
            return True                          # pass or grey-zone escalation possible
        if not check:
            return False
        C = arms.get(Nc, [])
        if len(C) < 2:
            return False
        sC = _lnz_stats(C)
        d = sC["mean"] - sR["mean"]
        sig = math.hypot(sR["err"], sC["err"])
        enough = sC["M"] >= max(4, min_runs // 2) and sR["M"] >= max(4, min_runs // 2)
        if check_rule == "r16":
            return bool(enough and N < max_nsteps and abs(d) - 2 * sig > 2 * lz_tol)
        return bool(enough and N < max_nsteps and abs(d) > 2 * lz_tol and abs(d) > 2.5 * sig)

    def evaluate(N, final=False):
        Nc = max(1, N // 2)
        R = arms.setdefault(N, [])
        C = arms.setdefault(Nc, []) if check else []
        sR = _lnz_stats(R)
        Rg = sR["good"]
        lzR, _ = merge_weights(Rg)
        ks, j = split_half_ks(Rg, seed=len(R), post_map=post_map)
        # the pooled insertion test is reported only (no power, Round 9): computed for the final evaluation only
        insp = pooled_insertion(Rg) if final else (np.nan, np.nan, np.nan)
        h = dict(nsteps=N, runs=sR["M_all"], runs_good=sR["M"], n_failed=sR["n_failed"], wall=time.perf_counter() - t0,
                 logz=lzR, logz_meanlnz=sR["mean"], logz_err=sR["err"], logz_sd=sR["sd"], logz_mad_sd=sR["mad_sd"],
                 quoted=float(np.mean([r["logzerr"] for r in R]) / math.sqrt(sR["M"])), ks_split=ks, ks_param=j,
                 insertion_p_min=min(r["insertion"]["p"] for r in R), final=final,
                 **dict(zip(("ins_pooled_D", "ins_pooled_p", "ins_fisher_p"), insp)),
                 ncall=int(ncall_tot[0]), ncall_result=int(sum(r["ncall"] for r in R)))
        ok_bias, escalate = True, False
        if check:
            sC = _lnz_stats(C) if len(C) >= 2 else dict(M=len(C), mean=np.nan, err=np.inf, good=C, n_failed=0)
            d = sC["mean"] - sR["mean"]
            sig = math.hypot(sR["err"], sC["err"])
            ksa, ja = _ks_sets(sC["good"], Rg, post_map, seed=len(R) + 7) if len(sC["good"]) >= 2 else (np.nan, -1)
            h.update(check_nsteps=Nc, check_runs=sC["M"], dlnz=d, dlnz_err=sig, ks_arms=ksa, ks_arms_param=ja)
            enough = sC["M"] >= max(4, min_runs // 2) and sR["M"] >= max(4, min_runs // 2)
            if check_rule == "r16":
                need = bool(np.isfinite(ksa) and ksa > ks_tol)
                if need:
                    nl = ks_null(sC["good"], Rg, post_map, seed=len(R) + 101)
                    med, hi = float(np.median(nl)), float(nl.max())
                else:
                    med = hi = 0.0
                h.update(ks_null_med=med, ks_null_hi=hi)
                ks_pass = (not need) or (ksa * ksa - med * med <= ks_tol * ks_tol)
                ks_esc = need and (ksa * ksa - hi * hi > ks_tol * ks_tol)
                k_sig = 3.0 if min(sR["M"], sC["M"]) < min_runs else 2.0
                dl_esc = abs(d) - k_sig * sig > 2 * lz_tol
                ok_bias = bool(abs(d) + sig <= 2 * lz_tol and ks_pass)
                # excess scatter: per-run lnZ sd > 3.5 x the quoted sqrt(H/n) -> the chains do not mix (e.g. runs that
                # miss modes); longer chains help (SBC draw 73: ratio ~5, logZ 1.2 too low at 64 steps; Pandora suite
                # 1.2-2.5). Needs >= max(16, 2 min_runs) runs (the sd of lnZ from few runs is heavy-tailed).
                q_run = float(np.mean([r["logzerr"] for r in Rg]))
                # only if the scatter falls with the chain length (the check arm at N/2 scatters >= 1.25x more):
                # scatter from multimodality (e.g. a moon model fitted to a light curve without a moon, Round 18:
                # 4 escalations to nsteps 1024 without effect) is not cured by longer chains
                sc_falls = bool(sC["M"] >= max(4, min_runs // 2) and np.isfinite(sC.get("sd", np.nan))
                                and sC["sd"] >= 1.25 * sR["sd"])
                sc_esc = bool(sR["M"] >= max(16, 2 * min_runs) and sR["sd"] > 3.5 * q_run and sc_falls)
                h.update(scatter_ratio=float(sR["sd"] / q_run) if q_run > 0 else np.nan)
                # mode problem (Round 20): heavy-tailed or nsteps-independent excess scatter in either arm
                def mode_problem(st):
                    if st["M"] < max(12, min_runs) or not np.isfinite(st.get("sd", np.nan)):
                        return False
                    q_ = float(np.mean([r["logzerr"] for r in st["good"]]))
                    tail = st["sd"] > 3.0 * max(st["mad_sd"], 1e-12)
                    big = st["M"] >= max(16, 2 * min_runs) and st["sd"] > 2.75 * q_ and not sc_falls
                    return bool(tail or big)
                mm = mode_problem(sR) or (len(C) >= 2 and mode_problem(sC))
                h.update(mode_problem=mm, nlive=nlive)
                if mm:
                    multimodal[0] = True
                    if nlive < max_nlive:
                        hist.append(h)
                        if verbose:
                            print(f"  fit: nsteps {N}: per-run lnZ sd {sR['sd']:.3f} (MAD-sd {sR['mad_sd']:.3f}, quoted "
                                  f"{q_run:.3f}), check arm sd {sC.get('sd', np.nan):.3f}: mode problem -> n_live "
                                  f"{nlive} -> {2 * nlive}", flush=True)
                        if log:
                            log(h)
                        return False, "nlive", sR["M_all"]
                    dl_esc = ks_esc = sc_esc = False      # at max_nlive: no escalation by mode-noise
                escalate = bool(enough and N < max_nsteps and (dl_esc or ks_esc or sc_esc))
            else:
                # bias at N ~ dlnZ(N/2, N) if it falls at least like 1/nsteps (known-Z tests, SAMPLING.md Round 8):
                # pass if its ~1 sigma upper bound is <= 2 lz_tol; escalate if it is significantly above that
                ok_bias = bool(abs(d) + sig <= 2 * lz_tol and ksa <= ks_tol)
                escalate = bool(enough and N < max_nsteps and
                                ((abs(d) > 2 * lz_tol and abs(d) > 2.5 * sig) or ksa > max(ks_tol, 2 * ks)))
        hist.append(h)
        if verbose:
            s = (f"  fit: nsteps {N:4d}: {sR['M']:3d} runs x {nlive} live: logZ {lzR:.3f} +- {sR['err']:.3f} "
                 f"(per-run sd {sR['sd']:.3f}, MAD-sd {sR['mad_sd']:.3f}, quoted {h['quoted']:.3f}; mean lnZ {sR['mean']:.3f}, "
                 f"log-mean-Z {sR['zmean']:.3f}, failed runs {sR['n_failed']}), "
                 f"split KS {ks:.3f} (p{j})")
            if check:
                s += (f" | check nsteps {Nc}: {h['check_runs']} runs, dlnZ {h['dlnz']:+.3f} +- {h['dlnz_err']:.3f}, "
                      f"KS arms {h['ks_arms']:.3f} (p{h['ks_arms_param']})")
            print(s + f", insertion p min {h['insertion_p_min']:.2g} (pooled D {h['ins_pooled_D']:.4f} p {h['ins_pooled_p']:.2g}, "
                  f"Fisher p {h['ins_fisher_p']:.2g}), {h['wall']:.1f} s" + (" [final]" if final else ""),
                  flush=True)
        if log:
            log(h)
        # more than 10 % failed runs: n_live per run too small to find the best mode reliably
        own_ok = bool(sR["M"] >= min_runs and ks <= ks_tol and sR["err"] <= lz_tol)
        passed = bool(own_ok and ok_bias and sR["n_failed"] <= 0.1 * sR["M_all"])
        grey_r8 = check_rule != "r16" and (h.get("dlnz_err", np.inf) <= 0.6 * lz_tol or h.get("ks_arms", 0) > ks_tol)
        grey_r16 = (check_rule == "r16" and h.get("dlnz_err", np.inf) <= 0.6 * lz_tol
                    and abs(h.get("dlnz", 0.0)) - h.get("dlnz_err", np.inf) > lz_tol)
        # r16 grey zone for the KS: the result arm meets its own criteria, both arms have >= 2 min_runs runs, and
        # the noise-corrected KS (median null) still exceeds ks_tol -> by the point estimate the arms differ by more
        # than the tolerance; doubling nsteps is cheaper than running to max_runs (SBC draws of Round 12 that
        # needed long chains stayed undecided without it, SAMPLING.md Round 16)
        if (check_rule == "r16" and check and h.get("ks_arms", 0) > ks_tol
                and h["ks_arms"] ** 2 - h.get("ks_null_med", 0.0) ** 2 > ks_tol ** 2
                and min(h.get("runs_good", 0), h.get("check_runs", 0)) >= 2 * min_runs):
            grey_r16 = True
        if check and own_ok and not ok_bias and N < max_nsteps and (grey_r8 or grey_r16) and not h.get("mode_problem"):
            # grey zone (C0: dlnZ ~ 0.15-0.2 or KS arms ~ ks_tol for 50+ runs): the result arm is converged by
            # its own criteria but the nsteps check does not pass, and the check is precise enough to say so
            # (C7: escalated at dlnZ 0.10 +- 0.10 without the sigma condition) -> doubling nsteps is cheaper
            escalate = True
        return passed, escalate, sR["M_all"]

    # continuous scheduling: the pool is kept full (no batch barrier); a new run goes to the arm with fewer
    # runs (done + in flight); the checks are evaluated after every `threads` completed runs (the workers
    # keep running meanwhile: the compiled runs release the GIL)
    from concurrent.futures import wait, FIRST_COMPLETED
    inflight = {}                                # future -> (nsteps, its own stop flag)
    passed = False
    n_aborted = [0]
    n_obsolete = [0]                             # runs aborted at an escalation (their arm was dropped)
    n_eval = [0, 0]                              # full evaluations: regular cadence / triggered by precheck
    t_eval = [0.0]                               # wall time of the Python checks (main thread)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        def top_up():
            nonlocal nxt
            while len(inflight) < threads:
                Nc = max(1, N // 2)
                nR = len(arms.get(N, [])) + sum(v[0] == N for v in inflight.values())
                nC = len(arms.get(Nc, [])) + sum(v[0] == Nc for v in inflight.values())
                ns = Nc if (check and nC < nR) else N
                while nxt in used:                # smallest free index (fills the gaps left by a crash)
                    nxt += 1
                idx = nxt
                used.add(idx)
                sd = seed + 1000 * idx
                flag = np.zeros(1, dtype=np.int64)
                inflight[ex.submit(run, ll_u, ndim, wrapped=wrapped, nlive=nlive, nsteps=ns, k=1, seed=sd, dirs=dirs,
                                   frac_remain=frac_remain, paxis=paxis, kpair=kpair, stop=flag, pglob=pglob,
                                   recluster=recluster, data=data, **(run_kw or {}))] = (ns, flag, idx)

        def escalate_nlive():
            nonlocal nlive, nlesc
            dropped_idx = []
            for ns_ in list(arms):
                dropped_idx += [int(r.get("idx", -1)) for r in arms.pop(ns_)]
            for f, (ns, flag, i_) in inflight.items():
                flag[0] = 1
                n_obsolete[0] += 1
                dropped_idx.append(int(i_))
            nlive *= 2
            nlesc += 1
            if callback is not None:
                callback("state", N=N, nesc=nesc, nlive=nlive, nlesc=nlesc, dropped=None, dropped_idx=dropped_idx)

        def escalate_now():
            nonlocal N, nesc
            if verbose:
                print(f"  fit: nsteps {N} vs {max(1, N // 2)} disagree -> nsteps {2 * N}", flush=True)
            dropped = max(1, N // 2)
            dropped_idx = [int(r.get("idx", -1)) for r in arms.pop(dropped, [])]
            N *= 2
            nesc += 1
            # runs of the dropped arm (old N / 2) still in flight are useless: abort them now
            for f, (ns, flag, i_) in inflight.items():
                if ns != N and ns != max(1, N // 2):
                    flag[0] = 1
                    n_obsolete[0] += 1
                    dropped_idx.append(int(i_))
            if callback is not None:
                callback("state", N=N, nesc=nesc, nlive=nlive, nlesc=nlesc, dropped=dropped, dropped_idx=dropped_idx)

        # When to evaluate (Round 13): the full evaluation (merges, KS) costs ~0.5-3 s, so up to Round 12 it ran
        # after every `threads` finished runs, and a fit passed on average ~half a run time after its criteria
        # were met. Now the cheap scalar precheck runs after every finished run and triggers the full evaluation
        # as soon as a pass (or a dlnZ escalation) is possible; while the precheck stays true but the KS checks
        # fail, the full evaluation is repeated every max(1, threads // 4) runs. The regular cadence stays.
        stop_now = False
        if resume is not None and len(arms.get(N, [])) >= 2:
            # resumed: evaluate the stored runs first (a fit that had passed passes again without new runs)
            while True:
                passed, escalate, M = evaluate(N)
                if passed or M >= max_runs:
                    stop_now = True
                    break
                if not escalate:
                    break
                if escalate == "nlive":
                    escalate_nlive()
                else:
                    escalate_now()
                if len(arms.get(N, [])) < 2:
                    break
        if not stop_now:
            top_up()
        ndone = 0
        pre_last = False
        while not stop_now:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in done:
                ns, _flag, idx = inflight.pop(f)
                r = f.result()
                r.pop("samples_u", None)
                ncall_tot[0] += r["ncall"]
                if _flag[0] != 0:                # aborted (dropped arm): incomplete, never used
                    ndone += 1
                    continue
                if (ns == N or (check and ns == max(1, N // 2))) and r["nlive"] == nlive:
                    r["idx"] = int(idx)
                    if callback is not None:
                        callback("run", r=r, nsteps=ns, idx=idx)
                    # kept sorted by seed (Round 14): merges, split halves and KS depend only on WHICH runs are
                    # done, not on the order in which they finished (Round 13: KS arms 0.049 vs 0.053 for the same
                    # runs in a different order flipped an escalation at ks_tol 0.05)
                    arms.setdefault(ns, []).append(r)
                    arms[ns].sort(key=lambda q: q["seed"])
                ndone += 1
            top_up()                             # refill before the (slow, Python) evaluation
            if len(arms.get(N, [])) < 2:
                continue
            regular = ndone >= threads
            pre = precheck(N)
            if regular or (pre and (not pre_last or ndone >= max(1, threads // 4))):
                ndone = 0
                n_eval[0 if regular else 1] += 1
                te = time.perf_counter()
                passed, escalate, M = evaluate(N)
                t_eval[0] += time.perf_counter() - te
                pre_last = pre
                if passed or (M >= max_runs and escalate != "nlive"):
                    break
                if escalate == "nlive":
                    escalate_nlive()
                    pre_last = False
                elif escalate:
                    escalate_now()
                    pre_last = False
            top_up()
        # abort the runs still in flight (the compiled loop polls its stop flag every iteration) and discard them:
        # waiting for them cost up to one run time (C7 at nsteps 256: 140 s of 790)
        for ns, flag, _i in inflight.values():
            flag[0] = 1
        for f in list(inflight):
            inflight.pop(f)
            r = f.result()
            ncall_tot[0] += r["ncall"]
            n_aborted[0] += 1
    passed, _, _ = evaluate(N, final=True)
    bad = set(failed_runs(arms[N]))
    R = [r for i, r in enumerate(arms[N]) if i not in bad]
    lzm, samples, ess = merge(R, seed=seed)
    zg = np.array([r["logz"] for r in R])        # bootstrap of mean lnZ over runs (scalars; the full re-merge
    _rng = np.random.default_rng(seed)            # bootstrap_logz cost ~1 s per resample at 100 runs)
    lzb = float(np.std(zg[_rng.integers(0, len(zg), (1000, len(zg)))].mean(axis=1), ddof=1)) if len(zg) > 1 else np.nan
    mixm = np.mean([r["mix"] for r in R], axis=0)
    return dict(mix=mixm, logz=lzm, logzerr=hist[-1]["logz_err"], logzerr_boot=lzb, samples_u=samples, ess=ess,
                runs=R, n_failed=len(bad), n_aborted=n_aborted[0], n_obsolete=n_obsolete[0], n_eval=n_eval[0],
                n_eval_pre=n_eval[1], t_eval=t_eval[0], t_compile=t_compile, runs_check=arms.get(max(1, N // 2), []) if check else [], history=hist, passed=passed,
                wall=time.perf_counter() - t0, ncall=hist[-1]["ncall"], nlive=nlive, nlive0=nlive0,
                n_nlive_escalations=nlesc, multimodal=multimodal[0], nsteps=N, nsteps0=nsteps,
                n_escalations=nesc, dlnz_check=hist[-1].get("dlnz"), dlnz_check_err=hist[-1].get("dlnz_err"),
                ks_arms=hist[-1].get("ks_arms"), maxL=float(max(r["maxL"] for r in R)))
