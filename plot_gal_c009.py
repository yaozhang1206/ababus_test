"""Plot HOD-galaxy P0 (shot-noise subtracted) and P2: emulator vs Abacus (two velocity copies), two HOD variants."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = np.load("/work/hdd/bdne/yzhang116/abacus_c009/galaxies/pk_gal_c009.npy", allow_pickle=True).item()
hods = [h for h in ("neutral", "ext") if f"emu_{h}" in R]
fig, ax = plt.subplots(2, 2 * len(hods), figsize=(5.6 * 2 * len(hods), 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = ax.reshape(2, -1)
sty = {"abacus_custom": dict(c="k", lw=2, label="Abacus (custom vel)"),
       "abacus_nbodyconc": dict(c="0.5", ls="--", label="Abacus (nbodyconc vel + conc)"),
       "emu": dict(c="C0", label="emulator, Abacus phases")}
for j, h in enumerate(hods):
    for l, ell in enumerate(("p0", "p2")):
        a, r = ax[0, 2 * j + l], ax[1, 2 * j + l]
        ref = R[f"abacus_custom_{h}"]
        for c, st in sty.items():
            s = R.get(f"{c}_{h}")
            if s is None:
                continue
            k = s["k"]; m = np.isfinite(k) & (k < 0.6)
            p = s[ell] - (1 / s["nbar"] if ell == "p0" else 0)
            lab = f"{st['label']}: n={s['nbar']*1e4:.2f}e-4, fsat={s['fsat']:.3f}, Δ={s['info']['delta']:+.3f}"
            a.plot(k[m], k[m] * p[m], **dict(st, label=lab))
            pr = ref[ell] - (1 / ref["nbar"] if ell == "p0" else 0)
            if c != "abacus_custom":
                r.plot(k[m], p[m] / pr[m], c=st["c"], ls=st.get("ls", "-"), label=f"{st['label']} / Abacus (custom vel)")
        a.set_title(f"HOD '{h}': galaxy $P_{ell[1]}$" + (" (shot noise subtracted)" if ell == "p0" else ""), fontsize=10)
        a.set_ylabel(rf"$k P_{ell[1]}$"); a.legend(fontsize=7)
        r.axhline(1, c="k", lw=0.8); r.axhspan(0.98, 1.02, color="0.88", zorder=0)
        r.set_ylim(0.85, 1.15) if ell == "p0" else r.set_ylim(0.7, 1.3)
        r.set_xscale("log"); r.set_xlabel(r"$k$ [$h$/Mpc]"); r.legend(fontsize=7)
plt.tight_layout()
out = "/u/yzhang116/test_discodj/lh4000/abacus/gal_c009_hod_pk.png"
plt.savefig(out, dpi=120); print(out)
for h in hods:
    ref = R[f"abacus_custom_{h}"]; e = R[f"emu_{h}"]; v = R[f"abacus_nbodyconc_{h}"]
    print(f"--- HOD {h}")
    for kk in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        i = np.nanargmin(abs(ref["k"] - kk))
        q0 = lambda s: s["p0"][i] - 1 / s["nbar"]
        print(f"k={ref['k'][i]:.3f}  emu/ab P0 {q0(e)/q0(ref):.3f} P2 {e['p2'][i]/ref['p2'][i]:.3f}   "
              f"emu/ab_nbodyconc P0 {q0(e)/q0(v):.3f} P2 {e['p2'][i]/v['p2'][i]:.3f}")
