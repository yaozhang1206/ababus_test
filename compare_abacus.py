"""DISCO-DJ + transformer 2 Gpc/h catalog vs AbacusSummit_base_c009_ph000 CompaSO halos, z = 0.5, M > 5e12.

HMF, redshift-space P0 and P2 (stats_2gpc.pk_ell: CIC 512^3, RSD along each axis, 3-axis mean, no shot-noise
subtraction) and the real-space halo cross-correlation coefficient r(k) (phase-matching check).
Gaussian error bands are for one 2 Gpc/h box; with matched phases the ratio scatter is smaller than that.

    python compare_abacus.py --emu CAT.npy --out stats.npz --fig fig.png
Env: conda pm (Pylians, h5py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats_2gpc as S2      # noqa: E402

L, OM, Z, MMIN = 2000.0, 0.3137721, 0.5, 5e12
ABACUS = {"custom": "/work/hdd/bdne/maho3/cmass-ili/abacus/custom/L2000-N256/30/halos.h5",
          "nbodyconc": "/work/hdd/bdne/maho3/cmass-ili/abacus/nbodyconc/L2000-N256/30/halos.h5"}


def load_abacus(path):
    import h5py
    with h5py.File(path) as f:
        g = f["0.666667"]
        cat = np.column_stack([g["pos"][:], 10.0 ** g["mass"][:].astype(np.float64), g["vel"][:]])
    return cat[cat[:, 3] > MMIN]


def load_emu(path):
    cat = np.load(path)[:, :7].astype(np.float64)
    return cat[cat[:, 3] > MMIN]


def stats(cat, ng=512):
    k, p, nm = S2.pk_ell(cat, OM, L, ng)
    h, n = S2.hmf(cat, L)
    return dict(n=len(cat), hmf=h, counts=n, k=k, p0=p[:, 0], p2=p[:, 1], nmodes=nm, nbar=len(cat) / L ** 3)


def cross_r(a, b, ng=256, nb=40, kmax=0.4):
    import MAS_library as MASL
    out = []
    for c in (a, b):
        d = np.zeros((ng,) * 3, np.float32)
        MASL.MA(np.ascontiguousarray(c[:, :3], np.float32), d, L, "CIC", verbose=False)
        d /= d.mean(dtype=np.float64)
        d -= 1
        out.append(np.fft.rfftn(d))
    k1 = np.fft.fftfreq(ng, 1.0 / ng)
    kz = np.arange(ng // 2 + 1)
    kk = (np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2) * 2 * np.pi / L).ravel()
    w = np.broadcast_to(np.where(kz == 0, 1.0, 2.0)[None, None, :], (ng, ng, ng // 2 + 1)).ravel()
    edges = np.linspace(0, kmax, nb + 1)
    i = np.digitize(kk, edges) - 1
    ok = (i >= 0) & (i < nb)
    A, B = out[0].ravel()[ok], out[1].ravel()[ok]
    i, w = i[ok], w[ok]
    ab = np.bincount(i, w * (A * np.conj(B)).real, nb)
    aa = np.bincount(i, w * np.abs(A) ** 2, nb)
    bb = np.bincount(i, w * np.abs(B) ** 2, nb)
    nmod = np.bincount(i, w, nb)
    return 0.5 * (edges[1:] + edges[:-1]), ab / np.sqrt(aa * bb), nmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emu", required=True)
    ap.add_argument("--emu-random", default=None, help="optional: same cosmology, unmatched phases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    cats = {"abacus": load_abacus(ABACUS["custom"]), "emu": load_emu(a.emu)}
    if a.emu_random:
        cats["emu_random"] = load_emu(a.emu_random)
    R = {n: stats(c) for n, c in cats.items()}
    R["abacus_nbodyconc_vel"] = stats(load_abacus(ABACUS["nbodyconc"]))
    kr, r, _ = cross_r(cats["emu"], cats["abacus"])
    R["xcorr"] = dict(k=kr, r=r)
    if a.emu_random:
        R["xcorr_random"] = dict(k=kr, r=cross_r(cats["emu_random"], cats["abacus"])[1])
    np.save(a.out, R, allow_pickle=True)
    ab, em = R["abacus"], R["emu"]
    print(f"n(M>5e12): abacus {ab['n']} emu {em['n']} ratio {em['n'] / ab['n']:.4f}")
    mc = 0.5 * (S2.MBINS[1:] + S2.MBINS[:-1])
    for j in range(len(mc)):
        if ab["counts"][j] > 0:
            print(f"  logM {mc[j]:.2f}: N abacus {ab['counts'][j]:8d} emu {em['counts'][j]:8d} "
                  f"ratio {em['counts'][j] / ab['counts'][j]:.3f} +- {np.sqrt(1 / max(em['counts'][j], 1) + 1 / ab['counts'][j]):.3f}")
    s = np.isfinite(ab["p0"]) & (ab["k"] < 0.3)
    for j in np.where(s)[0][::4]:
        print(f"  k {ab['k'][j]:.4f}: P0 emu/abacus {em['p0'][j] / ab['p0'][j]:.3f}  P2 emu/abacus "
              f"{em['p2'][j] / ab['p2'][j]:.3f}  r_hh {np.interp(ab['k'][j], kr, r):.3f}")
    plot(R, a.fig)


def plot(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mc = 0.5 * (S2.MBINS[1:] + S2.MBINS[:-1])
    sty = {"abacus": dict(c="k", label="AbacusSummit c009 ph000 (CompaSO)"),
           "emu": dict(c="C0", label="DISCO-DJ + transformer, Abacus phases"),
           "emu_random": dict(c="C2", ls=":", label="DISCO-DJ + transformer, random phases"),
           "abacus_nbodyconc_vel": dict(c="0.5", ls="--", lw=1, label="Abacus (nbodyconc velocities)")}
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex="col")
    ab = R["abacus"]
    for name, st in R.items():
        if name not in sty:
            continue
        ok = st["counts"] > 0
        if name != "abacus_nbodyconc_vel":
            e = st["hmf"] / np.sqrt(np.maximum(st["counts"], 1))
            ax[0, 0].errorbar(mc[ok], (st["hmf"] * 10 ** mc)[ok], yerr=(e * 10 ** mc)[ok], marker="o", ms=3, capsize=2, **sty[name])
            okb = ok & (ab["counts"] > 0)
            ax[1, 0].errorbar(mc[okb], st["counts"][okb] / ab["counts"][okb],
                              yerr=np.sqrt(1 / np.maximum(st["counts"], 1) + 1 / np.maximum(ab["counts"], 1))[okb],
                              marker="o", ms=3, capsize=2, c=sty[name]["c"], ls=sty[name].get("ls", "-"))
        s = np.isfinite(st["p0"])
        for j, key in enumerate(["p0", "p2"]):
            ax[0, 1 + j].plot(st["k"][s], st["k"][s] * st[key][s], **sty[name])
            ax[1, 1 + j].plot(st["k"][s], st[key][s] / ab[key][s], c=sty[name]["c"], ls=sty[name].get("ls", "-"),
                              lw=sty[name].get("lw", 1.5))
    e0, e2 = S2.gauss_err(ab)
    s = np.isfinite(ab["p0"])
    for j, (key, e) in enumerate([("p0", e0), ("p2", e2)]):
        ax[0, 1 + j].fill_between(ab["k"][s], (ab["k"] * (ab[key] - e))[s], (ab["k"] * (ab[key] + e))[s], color="k", alpha=0.15)
        ax[1, 1 + j].fill_between(ab["k"][s], (1 - e / np.abs(ab[key]))[s], (1 + e / np.abs(ab[key]))[s], color="k", alpha=0.15,
                                  label="Gaussian error, one 2 Gpc/h box")
    ax[0, 3].plot(R["xcorr"]["k"], R["xcorr"]["r"], c="C0", label="emu (Abacus phases) x Abacus")
    ax[0, 3].set_ylabel("$r_{hh}(k)$ real space"); ax[0, 3].legend(fontsize=7)
    ax[0, 3].set_xscale("log"); ax[0, 3].set_xlim(0.004, 0.42); ax[0, 3].set_ylim(0.75, 1.0)
    ax[0, 3].grid(alpha=0.3, which="both"); ax[0, 3].set_xlabel("k [h/Mpc]")
    ax[0, 3].tick_params(labelbottom=True)   # shared-x hides labels when ax[1, 3] is off
    ax[1, 3].axis("off")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_ylabel("M dn/dlogM [(h/Mpc)^3]"); ax[1, 0].set_ylabel("ratio to Abacus")
    ax[0, 1].set_ylabel("$k P_0$"); ax[0, 2].set_ylabel("$k P_2$")
    for j in (1, 2):
        ax[0, j].set_xscale("log"); ax[0, j].set_xlim(0.004, 0.5)
        ax[1, j].set_ylim(0.85, 1.15); ax[1, j].axhline(1, c="0.6", lw=0.8)
    ax[1, 0].set_ylim(0.7, 1.3); ax[1, 0].axhline(1, c="0.6", lw=0.8)
    ax[1, 0].set_xlabel("log10 M [Msun/h]"); ax[1, 1].set_xlabel("k [h/Mpc]"); ax[1, 2].set_xlabel("k [h/Mpc]")
    ax[0, 3].set_xlabel("k [h/Mpc]")
    ax[0, 0].legend(fontsize=7); ax[1, 1].legend(fontsize=7)
    fig.suptitle("z = 0.5, M > 5e12 Msun/h, 2 Gpc/h: AbacusSummit c009 (massless nu) vs DISCO-DJ + transformer")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
