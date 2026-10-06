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

acos_poly(x) (P16) is a branch-free arccos for |x| <= 1 (fdlibm's e_acos
with selects instead of branches), so loops that call it vectorise too;
<= 1 ulp from libm, ~0.9 instead of ~8.4 ns per element.
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


# fdlibm e_asin / e_acos rational kernel R(z) ~ (asin(sqrt z) / sqrt z - 1) / z
PS0 = 1.66666666666666657415e-01
PS1 = -3.25565818622400915405e-01
PS2 = 2.01212532134862925881e-01
PS3 = -4.00555345006794114027e-02
PS4 = 7.91534994289814532176e-04
PS5 = 3.47933107596021167570e-05
QS1 = -2.40339491173441421878e+00
QS2 = 2.02094576023350569471e+00
QS3 = -6.88283971605453293030e-01
QS4 = 7.70381505559019352791e-02
HALF_PI = 1.5707963267948966
PI = 3.141592653589793


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def acos_poly(x):
    """arccos(x) for -1 <= x <= 1, branch-free (P16).

    fdlibm: |x| <= 1/2: pi/2 - (x + x R(x^2)); |x| > 1/2: with
    z = (1 - |x|)/2 (exact), s = sqrt(z): acos(|x|) = 2 (s + s R(z)) and
    acos(-|x|) = pi - acos(|x|). Both branches are evaluated and selected,
    so LLVM vectorises loops that call it. <= 1 ulp from libm on [-1, 1],
    incl. +-1, +-1/2, 0 (test_acos_poly). The caller must clamp the
    argument to [-1, 1]; outside it the result is meaningless (NaN or
    garbage under fastmath)."""
    ax = abs(x)
    small = ax <= 0.5
    z = x * x if small else 0.5 * (1.0 - ax)
    s = np.sqrt(z)
    p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))))
    q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)))
    r = p / q
    a_small = HALF_PI - (x + x * r)
    t = 2.0 * (s + s * r)
    a_big = t if x > 0 else PI - t
    return a_small if small else a_big
