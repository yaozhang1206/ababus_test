"""AbacusSummit (zeldovich-PLT ZD_Version = 2) white-noise phases on a coarse N^3 grid, in numpy.

Re-implementation of zeldovich-PLT LoadPlane / PowerSpectrum::cgauss<2> (github.com/abacusorg/zeldovich-PLT,
src/zeldovich.cpp, src/power_spectrum.cpp, include/pcg-rng):
  * RNG: pcg64 = setseq_xsl_rr_128_64 with the default multiplier / increment and output of the *new* state,
    seeded as pcg64(seed): state = bump(seed + inc).  numpy's PCG64 (XSL-RR) is the same generator.
  * Logically one stream fills a MAX_PPD^3 (65536^3) Fourier cube: y-plane j (ky = j >= 0) starts at
    2 * j * MAX_PPD^2 draws; inside a plane, row z (kz wrapped into [-N/2+1, N/2]) and column x occupy
    logical position  row(z) * MAX_PPD + col(x),  row/col(i) = i for i <= N/2 else i + MAX_PPD - N.
    Each position consumes 2 draws (R, theta), Nyquist positions are skipped -> modes are independent of N.
  * cgauss<2>: u = (r + 1) * 2^-64 in (0, 1];  D = sqrt(-P ln u1) * exp(2 pi i u2)  (here P = 1 -> E|D|^2 = 1).
  * Planes ky = 1..N/2-1 are drawn, ky < 0 is the Hermitian conjugate; in the ky = 0 plane the kz > 0 half and
    the kz = 0, kx >= 0 half-line are drawn values, the rest is conjugate.  Nyquist (|k_i| = N/2) and k = 0 are 0.
  * Real field  delta(q) = sum_k D(k) exp(+i k.q) on the lattice q = i * L / N, with physical (X, Y, Z) =
    (code z, code y, code x) (see white_noise_rfft; the
    AbacusSummit IC file convention puts lattice [0,0,0] at -L/2; the halo catalogs here are shifted to [0, L)).

    w = white_noise_rfft(seed, N)   # complex64 (N, N, N//2+1), numpy rfftn convention [kx, ky, kz], E|w|^2 = 1
"""
import os

import numpy as np

MAX_PPD = 65536
MULT = 0x2360ED051FC65DA44385DF649FCCF645
INC = 0x5851F42D4C957F2D14057B7EF767814F
M128 = (1 << 128) - 1


def seeded_state(seed):
    """pcg-cpp engine(itype state): state_ = bump(state + increment)."""
    return ((seed + INC) * MULT + INC) & M128


def make_gen(seed, ndraw_skip):
    g = np.random.PCG64()
    g.state = {"bit_generator": "PCG64", "state": {"state": seeded_state(seed), "inc": INC},
               "has_uint32": 0, "uinteger": 0}
    if ndraw_skip:
        g.advance(int(ndraw_skip))
    return g


def ref_draws(seed, n, skip=0):
    """Pure-python pcg64 (C++ semantics: bump then XSL-RR of the new state) for testing make_gen."""
    s = seeded_state(seed)
    for _ in range(skip):
        s = (s * MULT + INC) & M128
    out = []
    for _ in range(n):
        s = (s * MULT + INC) & M128
        hi, lo = s >> 64, s & ((1 << 64) - 1)
        x, rot = hi ^ lo, hi >> 58
        out.append(((x >> rot) | (x << (64 - rot))) & ((1 << 64) - 1))
    return np.array(out, dtype=np.uint64)


def plane(seed, y, N):
    """Drawn (P = 1) modes of y-plane y (0 <= y < N/2) as complex128 array [z, x]; Nyquist rows/cols zeroed."""
    h = N // 2
    g = make_gen(seed, 2 * y * MAX_PPD * MAX_PPD)
    r = np.empty((N, 2 * N), np.uint64)        # per row: (R, theta) pairs for x = 0..N-1
    for z in range(N):
        if z == h + 1:
            g.advance(2 * (MAX_PPD - N) * MAX_PPD)
        r[z, :2 * (h + 1)] = g.random_raw(2 * (h + 1))         # x = 0..h (x = h is Nyquist: drawn here, zeroed below)
        g.advance(2 * (MAX_PPD - N))
        r[z, 2 * (h + 1):] = g.random_raw(2 * (N - h - 1))      # x = h+1..N-1
    # NB: positions of Nyquist elements are *skipped* (not drawn) by zeldovich-PLT, but each skip advances the
    # stream by exactly the 2 draws a mode would use, so drawing and discarding them is equivalent.
    u = (r.astype(np.float64) + 1.0) * 2.0 ** -64
    u[r == np.uint64(0xFFFFFFFFFFFFFFFF)] = 1.0
    u = u.reshape(N, N, 2)
    d = np.sqrt(-np.log(u[..., 0])) * np.exp(2j * np.pi * u[..., 1])
    d[h, :] = 0
    d[:, h] = 0
    if y == h:
        d[:] = 0
    return d


def _plane_worker(args):
    seed, y, N, path = args
    full = np.load(path, mmap_mode="r+")
    full[y] = plane(seed, y, N).astype(np.complex64)          # full[y, z, x]
    full.flush()
    return y


def white_noise_full(seed, N, tmp_path, nproc=64):
    """Complex64 Fourier cube D[ky, kz, kx] (index layout of zeldovich-PLT) with the Hermitian structure."""
    import multiprocessing as mp
    h = N // 2
    full = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.complex64, shape=(N, N, N))
    del full
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        for _ in pool.imap_unordered(_plane_worker, [(seed, y, N, tmp_path) for y in range(h)]):
            pass
    full = np.load(tmp_path, mmap_mode="r+")
    neg = (-np.arange(N)) % N                                 # index of -k
    # ky < 0 planes (y = N-1 .. h+1) are conj of D(-k) from planes 1..h-1
    for y in range(1, h):
        full[N - y] = np.conj(full[y][np.ix_(neg, neg)])
    full[h] = 0
    # ky = 0 plane: keep kz in [1, h-1] (all kx) and kz = 0 with kx in [0, h-1]; set the partners to conj
    p0 = np.array(full[0])
    for z in range(0, h):
        xs = np.arange(h) if z == 0 else np.arange(N)
        p0[neg[z], neg[xs]] = np.conj(p0[z, xs])
    p0[0, 0] = 0
    full[0] = p0
    return full


def white_noise_rfft(seed, N, tmp_path, nproc=64):
    """numpy rfftn-convention white noise w[kx, ky, kz] (kz in [0, N/2]), complex64, E|w|^2 = 1."""
    full = white_noise_full(seed, N, tmp_path, nproc)
    # Physical axes: the plane index ("y") is physical y, the row index ("z") is physical x and the fastest
    # index ("x") is physical z -- determined empirically by cross-correlating with the AbacusSummit halos
    # (build_ic.py check: r = 0.97 at k = 0.006 h/Mpc for this layout, ~0 for the [x<->z]-swapped one).
    w = np.ascontiguousarray(np.transpose(full[:, :, :N // 2 + 1], (1, 0, 2)))   # [y, z, x] -> [X, Y, Z]
    del full
    os.remove(tmp_path)
    return w


def _test_rng():
    for seed, skip in [(12321, 0), (12321, 37), (1, 1000)]:
        g = make_gen(seed, skip)
        a = g.random_raw(8)
        b = ref_draws(seed, 8, skip)
        assert np.array_equal(a, b), (seed, skip, a, b)
    print("pcg64 numpy == pure-python C++ semantics: OK")


def _test_hermitian(N=16, seed=12321):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        full = np.array(white_noise_full(seed, N, os.path.join(d, "t.npy"), nproc=2))
    D = np.transpose(full, (2, 0, 1))                         # [kx, ky, kz]
    x = np.fft.ifftn(D)
    print(f"N={N}: max |Im| / max |Re| of real field = {np.abs(x.imag).max() / np.abs(x.real).max():.2e}; "
          f"mean |D|^2 (nonzero) = {np.mean(np.abs(D[D != 0]) ** 2):.3f}")
    # mode consistency across N: the low-k modes of N=16 must equal those of N=32
    with tempfile.TemporaryDirectory() as d:
        f32 = np.transpose(np.array(white_noise_full(seed, 32, os.path.join(d, "t.npy"), nproc=2)), (2, 0, 1))
    ks = [k for k in range(-N // 2 + 1, N // 2)]
    i16 = [k % N for k in ks]
    i32 = [k % 32 for k in ks]
    a = D[np.ix_(i16, i16, i16)]
    b = f32[np.ix_(i32, i32, i32)]
    print(f"modes |k_i| < {N // 2}: N={N} vs N=32 max diff {np.abs(a - b).max():.2e} (max |D| {np.abs(a).max():.2f})")


if __name__ == "__main__":
    _test_rng()
    _test_hermitian()
