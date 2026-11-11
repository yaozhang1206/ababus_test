"""z = 0 linear delta(x) on 1024^3 for a 2000 Mpc/h box with the AbacusSummit_base_c009_ph000 phases.

White noise from abacus_phases (zeldovich-PLT v2 RNG, ZD_Seed = 12321), amplitude = CAMB (Quijote settings)
P(k) normalised to sigma8 = 0.811362 with 2LPTic's top-hat (ic_dj.normalised_table) -> same conventions as
ic_dj.delta_k_host, so the result is a drop-in for ic_worker output (run_2gpc.run_box).

Check: cross-correlation coefficient r(k) between this field (sharp-k cut to 256^3) and the CompaSO z = 0.5
halo number density of the real AbacusSummit box, for lattice offsets 0 and L/2 and two axis orders.

    python build_ic.py --out /work/hdd/bdne/yzhang116/abacus_c009/ic_c009_ph000_N1024.npy
Env: conda pm (numpy, scipy, h5py), no GPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.fft

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import abacus_phases as AP          # noqa: E402
import ic_dj                        # noqa: E402

SEED, N, L, SIGMA8 = 12321, 1024, 2000.0, 0.811362
PK = "/work/hdd/bdne/yzhang116/abacus_c009/Pk_mm_z=0.000.txt"
HALOS = "/work/hdd/bdne/maho3/cmass-ili/abacus/custom/L2000-N256/30/halos.h5"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def halo_field(ng, mmin=5e12):
    import h5py
    with h5py.File(HALOS) as f:
        g = f["0.666667"]
        lm = g["mass"][:]
        pos = g["pos"][:].astype(np.float64)
    pos = pos[lm > np.log10(mmin)]
    idx = np.floor(pos / L * ng).astype(np.int64) % ng          # NGP is enough for k << k_Ny
    d = np.bincount((idx[:, 0] * ng + idx[:, 1]) * ng + idx[:, 2], minlength=ng ** 3).reshape((ng,) * 3)
    d = d / d.mean() - 1.0
    return np.fft.rfftn(d), len(pos)


def rk(a, b, ng, nb=24):
    k1 = np.fft.fftfreq(ng, 1.0 / ng)
    kz = np.arange(ng // 2 + 1)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2) * 2 * np.pi / L
    edges = np.linspace(0, 0.3, nb + 1)
    i = np.digitize(kk.ravel(), edges) - 1
    ok = (i >= 0) & (i < nb)
    w = np.where(kz[None, None, :] == 0, 1.0, 2.0).repeat(ng, 0).repeat(ng, 1).ravel()[ok]
    i = i[ok]
    ab = np.bincount(i, w * (a.ravel()[ok] * np.conj(b.ravel()[ok])).real, nb)
    aa = np.bincount(i, w * np.abs(a.ravel()[ok]) ** 2, nb)
    bb = np.bincount(i, w * np.abs(b.ravel()[ok]) ** 2, nb)
    return 0.5 * (edges[1:] + edges[:-1]), ab / np.sqrt(aa * bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=64)
    ap.add_argument("--ng-check", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    w = AP.white_noise_rfft(SEED, N, "/dev/shm/abacus_wn_full.npy", nproc=a.nproc)
    nz = w != 0
    log(f"white noise {w.shape} {w.dtype}: mean |w|^2 = {np.mean(np.abs(w[nz]) ** 2):.5f} ({time.time() - t0:.0f} s)")
    k, p = ic_dj.normalised_table(PK, SIGMA8)
    kk = ic_dj.kmag_half(N, L)
    amp = np.sqrt(np.interp(kk, k, p).astype(np.float32) / np.float32(L ** 3)) * np.float32(N ** 3)
    del kk
    amp[0, 0, 0] = 0
    w *= amp
    del amp
    # low-k copy for the check (sharp cut |k_i| < ng/2)
    ng = a.ng_check
    ix = np.r_[0:ng // 2, N - ng // 2 + 1:N]
    lowk = w[np.ix_(ix, ix, np.arange(ng // 2 + 1))].astype(np.complex128)
    lowk = np.insert(lowk, ng // 2, 0, axis=0)
    lowk = np.insert(lowk, ng // 2, 0, axis=1)
    lowk[:, :, ng // 2] = 0
    d = scipy.fft.irfftn(w, s=(N,) * 3, workers=32, overwrite_x=True).astype(np.float32, copy=False)
    del w
    log(f"delta(x) z=0: std {d.std():.4f} ({time.time() - t0:.0f} s)")
    tmp = a.out + ".tmp.npy"
    np.save(tmp, d)
    os.replace(tmp, a.out)
    del d
    log(f"wrote {a.out}")

    hk, nh = halo_field(ng)
    kc, r0 = rk(lowk, hk, ng)
    q = np.fft.fftfreq(ng, 1.0 / ng)
    par = (-1.0) ** (q[:, None, None] + q[None, :, None] + np.arange(ng // 2 + 1)[None, None, :])
    _, rh = rk(lowk * par, hk, ng)                             # lattice offset L/2 on every axis
    # axis-reversed (z, y, x) ordering of the IC field
    lin_x = np.fft.irfftn(lowk, s=(ng,) * 3)
    _, rt = rk(np.fft.rfftn(np.ascontiguousarray(lin_x.transpose(2, 1, 0))), hk, ng)
    res = dict(n_halos=nh, k=kc.tolist(), r_offset0=r0.tolist(), r_offsetL2=rh.tolist(), r_transposed=rt.tolist())
    with open(a.out.replace(".npy", "_xcorr.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for i in range(0, len(kc), 2):
        log(f"k={kc[i]:.4f}: r(offset 0) {r0[i]:+.3f}  r(offset L/2) {rh[i]:+.3f}  r(transposed) {rt[i]:+.3f}")


if __name__ == "__main__":
    main()
