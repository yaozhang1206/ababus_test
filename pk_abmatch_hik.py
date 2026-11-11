"""Redshift-space P0/P2 to k = 0.6 h/Mpc at matched abundance (Abacus top-N, N = emulator count above 5e12).

Same estimator as stats_2gpc.pk_ell (CIC, Pylians compensation, RSD along each axis, 3-axis mean, Delta k = 2 pi/1000)
but on a 1024^3 mesh (k_Nyq = 1.61 h/Mpc) and binned to k = 0.6.
    python pk_abmatch_hik.py --out stats_c009_abmatch_hik.npy [--ng 1024]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_abacus as CA   # noqa: E402

D = "/work/hdd/bdne/yzhang116/abacus_c009/"


def pk_ell(cat, ng, kmax=0.6):
    import MAS_library as MASL, Pk_library as PKL, redshift_space_library as RSL
    hub = 100.0 * np.sqrt(CA.OM * (1 + CA.Z) ** 3 + 1 - CA.OM)
    res = []
    for axis in range(3):
        pos = np.ascontiguousarray(cat[:, :3], np.float32).copy()
        vel = np.ascontiguousarray(cat[:, 4:7], np.float32)
        RSL.pos_redshift_space(pos, vel, CA.L, hub, CA.Z, axis)
        d = np.zeros((ng,) * 3, np.float32)
        MASL.MA(pos, d, CA.L, "CIC", verbose=False)
        d /= d.mean(dtype=np.float64); d -= 1.0
        pk = PKL.Pk(d, CA.L, axis, "CIC", int(os.environ.get("OMP_NUM_THREADS", 16)), verbose=False)
        res.append(pk.Pk[:, :2]); del d
    k, nm, p = pk.k3D, pk.Nmodes3D, np.mean(res, 0)
    nb = int(np.ceil(kmax / (2 * np.pi / 1000) + 0.5))
    edges = np.arange(1, nb + 2) * 2 * np.pi / 1000.0 - np.pi / 1000.0
    out, kc, nmb = np.full((nb, 2), np.nan), np.full(nb, np.nan), np.zeros(nb)
    for i in range(nb):
        s = (k >= edges[i]) & (k < edges[i + 1])
        if s.any():
            w = nm[s]; out[i] = (p[s] * w[:, None]).sum(0) / w.sum(); kc[i] = (k[s] * w).sum() / w.sum(); nmb[i] = w.sum()
    return dict(k=kc, p0=out[:, 0], p2=out[:, 1], nmodes=nmb, n=len(cat), nbar=len(cat) / CA.L ** 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ng", type=int, default=1024)
    a = ap.parse_args()
    emu = CA.load_emu(D + "catalogs/catalog_2gpc_9009.npy")
    emr = CA.load_emu(D + "catalogs/catalog_2gpc_9010.npy")
    R = {}
    for name in ("custom", "nbodyconc"):
        ab = CA.load_abacus(CA.ABACUS[name])
        top = ab[np.argsort(ab[:, 3])[::-1][:len(emu)]]
        print(name, "abundance-matched threshold log10 M =", np.log10(top[:, 3].min()), len(top), flush=True)
        R["abacus" if name == "custom" else "abacus_nbodyconc_vel"] = pk_ell(top, a.ng)
    R["emu"] = pk_ell(emu, a.ng); R["emu_random"] = pk_ell(emr, a.ng)
    np.save(a.out, R, allow_pickle=True)
    ab, em = R["abacus"], R["emu"]
    for kk in [0.007, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        i = np.nanargmin(abs(ab["k"] - kk))
        print(f"k={ab['k'][i]:.3f}  P0 {em['p0'][i] / ab['p0'][i]:.3f}  P2 {em['p2'][i] / ab['p2'][i]:.3f}  "
              f"(nbodyconc-vel Abacus: P0 {em['p0'][i] / R['abacus_nbodyconc_vel']['p0'][i]:.3f} "
              f"P2 {em['p2'][i] / R['abacus_nbodyconc_vel']['p2'][i]:.3f})", flush=True)


if __name__ == "__main__":
    main()
