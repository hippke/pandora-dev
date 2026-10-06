"""Vectorisable sine and cosine of a phase given in turns.

numba cannot vectorise loops that call libm sin/cos (no SVML in the PyPI
llvmlite), and scalar libm sin + cos cost ~8 ns per sample. sincos_turns()
is pure arithmetic (floor, multiply, add, select), so LLVM auto-vectorises
loops that call it (AVX2 / NEON) under fastmath.

The phase is reduced in turns before converting to radians:
    f = u - round(u)          in [-1/2, 1/2]   (exact)
    q = round(4 f)            quadrant, in {-2, ..., 2}
    d = f - q/4               in [-1/8, 1/8]   (exact: Sterbenz)
    r = 2 pi d                in [-pi/4, pi/4] (one rounding)
then fdlibm/Cephes minimax polynomials on [-pi/4, pi/4] and a quadrant
select. The error of sin(2 pi u), cos(2 pi u) is a few 1e-16 for any u and
independent of |u| (libm on 2 pi u in radians inherits the rounding of
2 pi u, i.e. the argument's ulp, which grows with |u|).
NaN or inf input gives NaN.
"""
import numpy as np
from numba import jit

TWO_PI = 6.283185307179586

# fdlibm __kernel_sin / __kernel_cos coefficients on [-pi/4, pi/4]
S1 = -1.66666666666666324348e-01
S2 = 8.33333333332248946124e-03
S3 = -1.98412698298579493134e-04
S4 = 2.75573137070700676789e-06
S5 = -2.50507602534068634195e-08
S6 = 1.58969099521155010221e-10
C1 = 4.16666666666666019037e-02
C2 = -1.38888888888741095749e-03
C3 = 2.48015872894767294178e-05
C4 = -2.75573143513906633035e-07
C5 = 2.08757232129817482790e-09
C6 = -1.13596475577881948265e-11


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def sincos_turns(u):
    """(sin(2 pi u), cos(2 pi u)) for a phase u in turns (any size)."""
    f = u - np.floor(u + 0.5)
    q = np.floor(4.0 * f + 0.5)
    r = TWO_PI * (f - 0.25 * q)
    z = r * r
    s = r + r * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))))
    c = 1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))))
    k = np.int64(q) & 3
    if (k & 1) == 1:
        s, c = c, s
    if k == 1 or k == 2:
        c = -c
    if k >= 2:
        s = -s
    return s, c
