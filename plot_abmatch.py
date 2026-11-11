"""P0/P2 at matched abundance: Abacus top-N (N = emulator count) vs emulator (Abacus phases) and random-phase control."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/work/hdd/bdne/yzhang116/abacus_c009/"
import sys
KMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
import os
if os.path.exists(D + "stats_c009_abmatch_hik.npy"):   # pk_abmatch_hik.py, 1024^3 mesh, k <= 0.6
    s = np.load(D + "stats_c009_abmatch_hik.npy", allow_pickle=True).item()
    ab, em, er, abv = s["abacus"], s["emu"], s["emu_random"], s["abacus_nbodyconc_vel"]
else:                                                   # compare_abacus.py, 512^3 mesh, k <= 0.5
    s = np.load(D + "stats_c009.npy", allow_pickle=True).item()
    ab = np.load(D + "stats_c009_abmatch.npy", allow_pickle=True).item()
    em, er, abv = s["emu"], s["emu_random"], None
k = ab["k"]; m = np.isfinite(k) & (k < KMAX)
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
for j, ell in enumerate(["p0", "p2"]):
    a = ax[0, j]
    a.plot(k[m], k[m] * ab[ell][m], "k-", lw=2, label=f"Abacus c009 top-N (N={ab['n']:,}, logM>12.765)")
    a.plot(k[m], k[m] * em[ell][m], "C0-", label=f"emulator, Abacus phases (N={em['n']:,})")
    a.plot(k[m], k[m] * er[ell][m], "C1--", lw=1, label=f"emulator, random phases (N={er['n']:,})")
    if abv is not None: a.plot(k[m], k[m] * abv[ell][m], "--", c="0.5", lw=1, label="Abacus top-N, nbodyconc velocities")
    if ell == "p0":
        a.plot(k[m], k[m] / ab["nbar"] * np.ones(m.sum()), "k:", lw=1, label="shot noise k/n̄")
    a.set_ylabel(rf"$k\,P_{ell[1]}(k)$  [$(h^{{-1}}{{\rm Mpc}})^2$]"); a.set_title(f"$P_{ell[1]}$, z = 0.5, redshift space, same number of halos")
    a.legend(fontsize=8)
    r = ax[1, j]
    r.plot(k[m], em[ell][m] / ab[ell][m], "C0-", label="emu (Abacus phases) / Abacus")
    r.plot(k[m], er[ell][m] / ab[ell][m], "C1--", lw=1, label="emu (random) / Abacus")
    if abv is not None: r.plot(k[m], em[ell][m] / abv[ell][m], ":", c="C0", lw=1.2, label="emu (Abacus phases) / Abacus nbodyconc-vel")
    r.axhline(1, c="k", lw=0.8); r.axhspan(0.99, 1.01, color="0.85", zorder=0)
    r.set_ylim(0.85, 1.15) if ell == "p0" else r.set_ylim(0.7, 1.2)
    r.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]"); r.set_ylabel("ratio"); r.legend(fontsize=8)
    r.set_xscale("log")
plt.tight_layout()
out = f"/u/yzhang116/test_discodj/lh4000/abacus/abacus_c009_abmatch_pk_k{KMAX:g}.png"
plt.savefig(out, dpi=130); print(out)
for kk in [0.007, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
    i = np.nanargmin(abs(k - kk))
    print(f"k={k[i]:.3f}  P0 {em['p0'][i]/ab['p0'][i]:.3f}  P2 {em['p2'][i]/ab['p2'][i]:.3f}   (random: {er['p0'][i]/ab['p0'][i]:.3f} {er['p2'][i]/ab['p2'][i]:.3f})")
