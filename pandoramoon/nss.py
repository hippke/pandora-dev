"""nss: a compiled nested slice sampler for Pandora (experimental).

Description: NSS_SAMPLER.md; development history and validation: SAMPLING.md (Rounds 4-8) in the
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
- Directions, dirs="cluster" (default since the Gaussian-mixture test,
  SAMPLING.md Round 4): the live points are clustered every
  `recluster` x n replacements (connected components of the K-nearest-
  neighbour graph, K = 2 ndim, neighbours among <= nbsub random reference
  points; new points inherit their start point's cluster). Each slice
  step uses the difference of two random live points of the start
  point's cluster (start excluded): local to the mode, but not to the
  position inside it. Clusters with < 2 other points fall back to "knn".
- Directions, dirs="localcov": every `recluster` x n replacements the
  mean local covariance (average over R = 200 random live points of the
  covariance of their K nearest neighbours) is computed; each slice step
  uses v = chol @ N(0, I). Local (within-mode) shape and scale, but the
  same for every chain, so no dependence on the start point's position.
- Directions, dirs="knnmh": knn directions made exact. N(x) = the K
  nearest of a fixed (per chain) reference set to the current point x; a
  slice step along U_a - U_b (a, b in N(x)) is accepted only if a, b are
  also in N(x') (Metropolis-Hastings check of the direction probability;
  `njump_acc` then counts these rejections). One knn search per step.
- Directions, dirs="pairs" (Round 7): v = +-(U_a - U_b) with a a random live point and b one of
  its `kpair` nearest neighbours (neighbour lists recomputed every `recluster` x n replacements, since
  Round 13 by pair_lists(): the same lists ~10x cheaper than the full K-nearest graph of cluster()).
  Local orientation and scale, but independent of the chain's position (exact).
- Directions, dirs="knn" (first version; biased for short chains, see
  SAMPLING.md): at the start
  of a chain, the K nearest live neighbours of the start point (distance
  scaled by the live points' per-dimension spread) are found; each slice
  step uses the difference of two random members of {start, neighbours}
  as direction vector (local orientation and scale; a chain stays in its
  mode). The direction distribution is fixed for the chain, so each step
  is a valid slice update of the constrained prior. For n > `nbsub`
  (default 2000) the neighbours are searched among nbsub random alive
  points (O(nbsub ndim) per chain).
- Optional mode jumps (`pjump`): with probability pjump per step an extra
  differential-evolution move x + (U_a - U_b) with two random live points
  (ter Braak 2006), accepted if L > Lmin (symmetric proposal while the
  live set is fixed during the chain). For comb-like posteriors (modes =
  shifted copies) it moves points between modes in proportion to their
  volume instead of their live counts.
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
MAXEXP = 100     # max linear step-outs per side and slice


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
    def pair_lists(U, alive, sd, kk, M, nbr, ref):
        """dirs="pairs" (Round 13): nbr[q, :kk] = the kk nearest alive points to U[q] (metric scaled by sd) among
        the reference set (all alive points if n <= M, else M distinct random ones), ascending. The same lists as
        the first kk columns of cluster()'s K-nearest graph (checked: test_knnfast.py), without the union-find
        (the labels are not used by "pairs") and faster: scaled reference coordinates stored transposed, so the
        squared distances of one point to all references are summed dimension by dimension over contiguous
        arrays (SIMD), then one selection pass keeps the kk smallest; no allocation per point."""
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
    def local_cov(U, alive, sd, K, M, R, ref):
        """mean over R random alive points y of the covariance of y's K
        nearest neighbours (among <= M distinct random alive reference
        points): the typical *local* (within-mode) shape, independent of
        any chain's start point. Returns its Cholesky factor."""
        n, ndim = U.shape
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
        C = np.zeros((ndim, ndim))
        d2 = np.empty(nref)
        mu = np.empty(ndim)
        for _r in range(R):
            while True:
                y = np.random.randint(n)
                if alive[y]:
                    break
            for a in range(nref):
                r = ref[a]
                s = 0.0
                for i in range(ndim):
                    t = (U[r, i] - U[y, i]) / sd[i]
                    s += t * t
                d2[a] = s
            sel = k_smallest_all(d2, K)
            m = sel.shape[0]
            for i in range(ndim):
                mu[i] = 0.0
                for a in range(m):
                    mu[i] += U[ref[sel[a]], i]
                mu[i] /= m
            for a in range(m):
                r = ref[sel[a]]
                for i in range(ndim):
                    di = U[r, i] - mu[i]
                    for i2 in range(i + 1):
                        C[i, i2] += di * (U[r, i2] - mu[i2])
        for i in range(ndim):
            for i2 in range(i + 1):
                C[i, i2] /= R * (K - 1)
                C[i2, i] = C[i, i2]
        for i in range(ndim):
            C[i, i] += 1e-12 + 1e-9 * C[i, i]
        return np.linalg.cholesky(C)

    @njit
    def knn_point(x, U, R, nR, sd, K, d2):
        """indices (into U) of the K nearest of U[R[:nR]] to the point x"""
        ndim = U.shape[1]
        for a in range(nR):
            r = R[a]
            s = 0.0
            for i in range(ndim):
                t = (U[r, i] - x[i]) / sd[i]
                s += t * t
            d2[a] = s
        sel = k_smallest_all(d2[:nR], K)
        out = np.empty(sel.shape[0], dtype=np.int64)
        for a in range(sel.shape[0]):
            out[a] = R[sel[a]]
        return out

    @njit
    def chain_knnmh(U, j, L0, R, nR, sd, K, Lmin, nsteps, scale, wrapped, x, xn, d2, D):
        """slice chain with position-dependent knn directions made reversible
        by a Metropolis-Hastings check: N(x) = the K nearest reference points
        U[R] to x (R fixed during the chain); a step uses v = U[a] - U[b] with
        a pair from N(x) (uniform over ordered pairs); the slice point x' is
        accepted only if a and b are also in N(x') (then p_x(v) = p_x'(v),
        detailed balance), else the chain stays at x.
        Returns (L, ncalls, nexpand, n_mh_reject)."""
        ndim = U.shape[1]
        for i in range(ndim):
            x[i] = U[j, i]
        L = L0
        nc = 0
        nexp = 0
        nrej = 0
        N = knn_point(x, U, R, nR, sd, K, d2)
        m = N.shape[0]
        v = np.empty(ndim)
        for _ in range(nsteps):
            ia = np.random.randint(m)
            ib = np.random.randint(m - 1)
            if ib >= ia:
                ib += 1
            a = N[ia]
            b = N[ib]
            vv = 0.0
            for i in range(ndim):
                v[i] = U[a, i] - U[b, i]
                vv += v[i] * v[i]
            if vv == 0.0:
                continue
            left = -scale * np.random.uniform(0.0, 1.0)
            right = left + scale
            for _e in range(MAXEXP):
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * left
                if not wrap_inside(xn, wrapped):
                    break
                nc += 1
                if LL(xn, D) > Lmin:
                    left -= scale
                    nexp += 1
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
                else:
                    break
            Lx = -np.inf
            ok = False
            while True:
                t = np.random.uniform(left, right)
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * t
                if wrap_inside(xn, wrapped):
                    nc += 1
                    Lx = LL(xn, D)
                    if Lx > Lmin:
                        ok = True
                        break
                if t < 0.0:
                    left = t
                else:
                    right = t
                if right - left < 1e-14:
                    break
            if not ok:
                continue
            N2 = knn_point(xn, U, R, nR, sd, K, d2)
            fa = False
            fb = False
            for q in range(N2.shape[0]):
                if N2[q] == a:
                    fa = True
                if N2[q] == b:
                    fb = True
            if fa and fb:
                for i in range(ndim):
                    x[i] = xn[i]
                L = Lx
                N = N2
                m = N.shape[0]
            else:
                nrej += 1
        return L, nc, nexp, nrej

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
            Lq = np.linalg.cholesky(C)
            for i in range(ndim):
                for i2 in range(ndim):
                    Lp[q, i, i2] = Lq[i, i2]

    @njit
    def chain(U, j, L0, nb, Lmin, nsteps, scale, wrapped, x, xn, alive, pjump, chol, dmode, paxis, sdv, nbr, kpair,
              pglob, D, Lp):
        """slice chain at fixed Lmin from U[j] (logL L0); directions =
        differences of members of nb; with probability pjump per step an
        additional differential-evolution jump x + (U_a - U_b) (a, b random
        alive live points; symmetric -> Metropolis accept if L > Lmin).
        Returns (L, ncalls, nexpand, njump_acc) with the end point in x"""
        ndim = U.shape[1]
        n = U.shape[0]
        for i in range(ndim):
            x[i] = U[j, i]
        L = L0
        nc = 0
        nexp = 0
        nja = 0
        m = nb.shape[0]
        v = np.empty(ndim)
        for _ in range(nsteps):
            if pjump > 0.0 and np.random.uniform(0.0, 1.0) < pjump:
                while True:
                    a = np.random.randint(n)
                    if alive[a]:
                        break
                while True:
                    b = np.random.randint(n)
                    if alive[b] and b != a:
                        break
                for i in range(ndim):
                    xn[i] = x[i] + U[a, i] - U[b, i]
                if wrap_inside(xn, wrapped):
                    nc += 1
                    Lx = LL(xn, D)
                    if Lx > Lmin:
                        for i in range(ndim):
                            x[i] = xn[i]
                        L = Lx
                        nja += 1
            a = np.random.randint(m)
            b = np.random.randint(m - 1)
            if b >= a:
                b += 1
            vv = 0.0
            ur = np.random.uniform(0.0, 1.0) if (paxis > 0.0 or pglob > 0.0) else 2.0
            if ur < paxis:
                # coordinate-axis slice (Gibbs-like), length ~ the live points' spread in that axis
                for i in range(ndim):
                    v[i] = 0.0
                ka = np.random.randint(ndim)
                v[ka] = 1.4142135623730951 * sdv[ka]
                vv = v[ka] * v[ka]
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
            elif dmode == 2:
                # local pair difference: a random alive point and one of its kpair nearest neighbours
                # (neighbour lists from the last reclustering; independent of the chain's position)
                while True:
                    pa = np.random.randint(n)
                    if alive[pa]:
                        break
                pb = nbr[pa, np.random.randint(kpair)]
                sg = 1.0 if np.random.uniform(0.0, 1.0) < 0.5 else -1.0
                for i in range(ndim):
                    v[i] = sg * (U[pa, i] - U[pb, i])
                    vv += v[i] * v[i]
            elif dmode == 1:
                z = np.random.standard_normal(ndim)
                for i in range(ndim):
                    s_ = 0.0
                    for i2 in range(i + 1):
                        s_ += chol[i, i2] * z[i2]
                    v[i] = s_
                    vv += s_ * s_
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
                    break
        return L, nc, nexp, nja

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
    def ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed, scale0, pjump, M, dirmode, recluster, R,
           paxis, kpair, stop, pglob, D, knnfast):
        n, ndim = U.shape
        # seeds (per thread in parallel mode: numba seeds each thread's state)
        np.random.seed(seed)
        cap = 1 << 16
        dU = np.empty((cap, ndim))
        dL = np.empty(cap)
        dlogw = np.empty(cap)
        nd = 0
        logX = 0.0
        logZ = -np.inf
        ncall = 0
        scale = scale0
        it = 0
        nexp_tot = 0
        nstep_tot = 0
        sd = np.empty(ndim)
        newU = np.empty((k, ndim))
        newL = np.empty(k)
        ncs = np.zeros(k, dtype=np.int64)
        nexps = np.zeros(k, dtype=np.int64)
        njs = np.zeros(k, dtype=np.int64)
        nj_tot = 0
        starts = np.empty(k, dtype=np.int64)
        nbs = np.empty((k, K + 1), dtype=np.int64)
        xs = np.empty((k, ndim))
        xns = np.empty((k, ndim))
        alive = np.ones(n, dtype=np.bool_)
        d2s = np.empty((k, n))
        idxs = np.empty((k, n), dtype=np.int64)
        sd_every = max(1, n // (10 * k))
        labels = np.zeros(n, dtype=np.int64)
        nbr = np.empty((n, K), dtype=np.int64)
        Lp = np.zeros((n if dirmode == 5 else 1, ndim, ndim))
        for q0 in range(n):
            for t0 in range(K):
                nbr[q0, t0] = q0
        ref = np.empty(n, dtype=np.int64)
        mcount = np.zeros(k, dtype=np.int64)
        nclus = 1
        nclus_max = 1
        nrecl = 0
        repl_since = n * 10           # force clustering at the first iteration
        chol = np.eye(ndim)
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
            if dirmode == 4 and knnfast != 0 and repl_since >= recluster * n:
                pair_lists(U, alive, sd, kpair, M, nbr, ref)
                nrecl += 1
                repl_since = 0
            if (dirmode == 1 or (dirmode == 4 and knnfast == 0) or dirmode == 5) and repl_since >= recluster * n:
                nclus = cluster(U, alive, sd, K, M, labels, nbr, ref)
                if dirmode == 5:
                    local_chols(U, alive, nbr, wrapped, Lp)
                nclus_max = max(nclus_max, nclus)
                nrecl += 1
                repl_since = 0
            if dirmode == 2 and repl_since >= recluster * n:
                chol = local_cov(U, alive, sd, K, M, R, ref)
                nrecl += 1
                repl_since = 0
            repl_since += k
            for c in range(k):
                while True:
                    j = np.random.randint(n)
                    if alive[j]:
                        break
                starts[c] = j
            for c in prange(k):
                m = 0
                if dirmode == 1:
                    m = members(starts[c], alive, labels, idxs[c])
                if dirmode == 2 or dirmode == 3 or dirmode == 4 or dirmode == 5:
                    mem = idxs[c, :2]
                    mcount[c] = 0
                elif m >= 2:
                    mem = idxs[c, :m]
                    mcount[c] = 0
                else:
                    mem = neighbours(U, starts[c], alive, sd, K, d2s[c], M, idxs[c])
                    mcount[c] = 1 if dirmode == 1 else 0
                if dirmode == 3:
                    nR = refset(alive, starts[c], M, idxs[c])
                    Lc, nc, ne, nj = chain_knnmh(U, starts[c], L[starts[c]], idxs[c], nR, sd, K, Lmin, nsteps, scale,
                                                 wrapped, xs[c], xns[c], d2s[c], D)
                else:
                    Lc, nc, ne, nj = chain(U, starts[c], L[starts[c]], mem, Lmin, nsteps, scale, wrapped, xs[c],
                                           xns[c], alive, pjump, chol,
                                           1 if dirmode == 2 else (2 if dirmode == 4 else (3 if dirmode == 5 else 0)),
                                           paxis, sd, nbr, kpair, pglob, D, Lp)
                njs[c] = nj
                newL[c] = Lc
                ncs[c] = nc
                nexps[c] = ne
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
                nj_tot += njs[c]
            nstep_tot += k * nsteps
            # scale adaptation: aim at ~1 expansion per step (interval comparable to the slice)
            if nexps.sum() > k * nsteps:
                scale *= 1.1
            else:
                scale /= 1.1
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
        return (dU[:nd], dL[:nd], dlogw[:nd], ncall, it, scale, nexp_tot / max(nstep_tot, 1), nj_tot,
                nclus, nclus_max, nrecl, nfallback, ins[:nins], mix / max(nmix, 1))

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
        max_iter=10_000_000, seed=1, scale0=1.0, threads=None, pjump=0.0, nbsub=2000, dirs="cluster",
        recluster=0.25, R=200, paxis=0.0, kpair=4, stop=None, pglob=0.0, data=None, knnfast=True):
    """returns dict(logz, logzerr, ncall, niter, wall, samples_u (equal weight), dead_u, dead_L, logw).
    data: None -> ll_u(u); a tuple D -> ll_u(u, D) (one compiled sampler for many data sets).
    knnfast (dirs="pairs" only): neighbour lists by pair_lists() (Round 13) instead of the full K-nearest graph +
    union-find of cluster() (the same lists up to rounding ties; False = the Round 7-12 code, for A/B tests)"""
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
    U, L = init(nlive, ndim, seed, D)
    t1 = time.perf_counter()
    assert dirs in ("cluster", "knn", "localcov", "knnmh", "pairs", "lcov")
    (dU, dL, dlogw, ncall, it, scale, exp_rate, njump, nclus, nclus_max, nrecl,
     nfallback, ins, mix) = ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed + 1, scale0, pjump, nbsub,
                          {"knn": 0, "cluster": 1, "localcov": 2, "knnmh": 3, "pairs": 4, "lcov": 5}[dirs], recluster, R,
                          paxis, kpair, stop if stop is not None else np.zeros(1, dtype=np.int64), pglob, D,
                          1 if knnfast else 0)
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
    return dict(nlive=int(nlive), ins_u=((ins + np.random.default_rng(seed + 11).uniform(size=len(ins))) / (nlive - k + 1)
                                       ).astype(np.float32), mix=mix, insertion=insd, logz=logz, logzerr=math.sqrt(max(H, 0.0) / nlive), H=H, ncall=int(ncall) + nlive, niter=int(it),
                wall=t2 - t0, wall_init=t1 - t0, samples_u=samples, ess=ess, dead_u=dU, dead_L=dL, logw=dlogw,
                maxL=float(dL.max()), final_scale=scale, expand_rate=exp_rate, njump_acc=int(njump),
                nclusters=int(nclus), nclusters_max=int(nclus_max), nrecluster=int(nrecl),
                nfallback=int(nfallback), dirs=dirs,
                jump_acc_per_chain=njump / max(it * k, 1))


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
    lx = -np.cumsum(dlx)
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
        accuracy=None):
    """Nested sampling with built-in convergence checks (SAMPLING.md Rounds 6b, 8).

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
    passed, nsteps, dlnz_check, ...)."""
    from concurrent.futures import ThreadPoolExecutor
    if accuracy is not None:
        a = ACCURACY[accuracy]
        nlive, lz_tol, ks_tol = a["nlive"], a["lz_tol"], a["ks_tol"]
        min_runs = min_runs or a["min_runs"]
    nsteps = nsteps or 4 * ndim
    max_nsteps = max_nsteps or 64 * ndim
    min_runs = min_runs or max(threads, 4)
    tc = time.perf_counter()
    make_sampler(ll_u, False, data is not None)  # compile once, outside the threads
    run(ll_u, ndim, wrapped=wrapped, nlive=max(40, 2 * ndim + 8), nsteps=2, k=1, max_iter=3, seed=seed + 999, data=data,
        dirs=dirs, kpair=min(kpair, 4))
    t_compile = time.perf_counter() - tc         # numba compile (first call per process); not in `wall`
    arms = {}                                    # nsteps -> list of runs
    hist = []
    t0 = time.perf_counter()
    nxt = 0
    N = nsteps
    nesc = 0
    ncall_tot = [0]

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
            # bias at N ~ dlnZ(N/2, N) if it falls at least like 1/nsteps (known-Z tests, SAMPLING.md Round 8):
            # pass if its ~1 sigma upper bound is <= 2 lz_tol; escalate if it is significantly above that
            ok_bias = bool(abs(d) + sig <= 2 * lz_tol and ksa <= ks_tol)
            enough = sC["M"] >= max(4, min_runs // 2) and sR["M"] >= max(4, min_runs // 2)
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
        if check and own_ok and not ok_bias and N < max_nsteps and (h["dlnz_err"] <= 0.6 * lz_tol
                                                                     or h["ks_arms"] > ks_tol):
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
                sd = seed + 1000 * nxt
                nxt += 1
                flag = np.zeros(1, dtype=np.int64)
                inflight[ex.submit(run, ll_u, ndim, wrapped=wrapped, nlive=nlive, nsteps=ns, k=1, seed=sd, dirs=dirs,
                                   frac_remain=frac_remain, paxis=paxis, kpair=kpair, stop=flag, pglob=pglob,
                                   recluster=recluster, data=data)] = (ns, flag)

        # When to evaluate (Round 13): the full evaluation (merges, KS) costs ~0.5-3 s, so up to Round 12 it ran
        # after every `threads` finished runs, and a fit passed on average ~half a run time after its criteria
        # were met. Now the cheap scalar precheck runs after every finished run and triggers the full evaluation
        # as soon as a pass (or a dlnZ escalation) is possible; while the precheck stays true but the KS checks
        # fail, the full evaluation is repeated every max(1, threads // 4) runs. The regular cadence stays.
        top_up()
        ndone = 0
        pre_last = False
        while True:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in done:
                ns, _flag = inflight.pop(f)
                r = f.result()
                r.pop("samples_u", None)
                ncall_tot[0] += r["ncall"]
                if ns == N or (check and ns == max(1, N // 2)):
                    arms.setdefault(ns, []).append(r)
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
                if passed or M >= max_runs:
                    break
                if escalate:
                    if verbose:
                        print(f"  fit: nsteps {N} vs {max(1, N // 2)} disagree -> nsteps {2 * N}", flush=True)
                    arms.pop(max(1, N // 2), None)
                    N *= 2
                    nesc += 1
                    pre_last = False
                    # runs of the dropped arm (old N / 2) still in flight are useless: abort them now
                    for f, (ns, flag) in inflight.items():
                        if ns != N and ns != max(1, N // 2):
                            flag[0] = 1
                            n_obsolete[0] += 1
            top_up()
        # abort the runs still in flight (the compiled loop polls its stop flag every iteration) and discard them:
        # waiting for them cost up to one run time (C7 at nsteps 256: 140 s of 790)
        for ns, flag in inflight.values():
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
                wall=time.perf_counter() - t0, ncall=hist[-1]["ncall"], nlive=nlive, nsteps=N, nsteps0=nsteps,
                n_escalations=nesc, dlnz_check=hist[-1].get("dlnz"), dlnz_check_err=hist[-1].get("dlnz_err"),
                ks_arms=hist[-1].get("ks_arms"), maxL=float(max(r["maxL"] for r in R)))
