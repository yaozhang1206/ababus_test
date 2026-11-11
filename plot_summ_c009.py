"""Compare summ.py-style statistics (summ_c009.py output): zPk ell=0,2 and zBk/zQk ell=0, emulator vs Abacus.
    python plot_summ_c009.py [--dir DIR] [--pairs EMU:ABACUS ...] [--out fig.png]
Default pairs: HOD galaxies, emulator vs Abacus (custom and nbodyconc velocities), neutral and ext HOD.
"""
import argparse, os
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/work/hdd/bdne/yzhang116/abacus_c009/galaxies/summ/")
ap.add_argument("--pairs", nargs="+", default=[f"gal_emu_{h}:gal_abacus_{v}_{h}" for h in ("neutral", "ext") for v in ("custom", "nbodyconc")])
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "summ_c009_cmp.png"))
a = ap.parse_args()


def rd(n):
    with h5py.File(os.path.join(a.dir, n + ".h5")) as f:
        return {k: f[k][...] for k in f}, dict(f.attrs)


pairs = [p.split(":") for p in a.pairs if all(os.path.exists(os.path.join(a.dir, x + ".h5")) for x in p.split(":"))]
fig, ax = plt.subplots(len(pairs), 4, figsize=(20, 3.2 * len(pairs)), squeeze=False)
for i, (e, b) in enumerate(pairs):
    E, Ea = rd(e); B, Ba = rd(b)
    k = E["zPk_k3D"]
    for j, ell in enumerate((0, 2)):
        x = ax[i, j]
        x.plot(k, E["zPk"][:, ell] / B["zPk"][:, ell], label=f"real+RSD ell={ell}")
        if ell == 0:
            x.plot(k, E["Pk"][:, 0] / B["Pk"][:, 0], "--", label="real space ell=0")
        x.axhline(1, c="k", lw=.8); x.axhspan(.98, 1.02, color="0.88", zorder=0)
        x.set_ylim(0.8, 1.2); x.set_xlabel("k [h/Mpc]"); x.set_title(f"P{ell}: {e} / {b}", fontsize=8); x.legend(fontsize=7)
    if "zBk" in E and "zBk" in B:
        for j, key in enumerate(("zBk", "zQk")):
            x = ax[i, 2 + j]
            x.plot(E[key][0] / B[key][0], ".", ms=3, label=f"{key} ell=0 (redshift space)")
            x.plot(E[key[1:]][0] / B[key[1:]][0], ".", ms=3, alpha=.5, label=f"{key[1:]} ell=0 (real space)")
            x.axhline(1, c="k", lw=.8); x.axhspan(.95, 1.05, color="0.88", zorder=0)
            x.set_ylim(0.7, 1.3); x.set_xlabel("triangle index"); x.set_title(f"{key} ratio, n = {Ea['nbar']:.2e} / {Ba['nbar']:.2e}", fontsize=8)
            x.legend(fontsize=7)
plt.tight_layout(); plt.savefig(a.out, dpi=110); print(a.out)
