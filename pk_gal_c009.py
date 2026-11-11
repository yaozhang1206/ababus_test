"""Redshift-space P0/P2 of the HOD galaxies (hod_c009.py), same estimator as pk_abmatch_hik.pk_ell
(CIC 1024^3 compensated, RSD along each axis, 3-axis mean, Delta k = 2 pi/1000, k <= 0.6).
    python pk_gal_c009.py   -> galaxies/pk_gal_c009.npy
Env: conda pm.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pk_abmatch_hik as P   # noqa: E402

G = "/work/hdd/bdne/yzhang116/abacus_c009/galaxies/"
R = {}
for f in sorted(glob.glob(G + "gal_*.npy")):
    name = os.path.basename(f)[4:-4]
    g = np.load(f).astype(np.float64)
    cat = np.column_stack([g[:, :3], np.ones(len(g)), g[:, 3:6]])      # pk_ell layout: pos, mass, vel
    R[name] = P.pk_ell(cat, 1024)
    R[name]["fsat"] = float(g[:, 6].mean())
    R[name]["info"] = json.load(open(f[:-4] + ".json"))
    print(name, len(g), f"n = {len(g) / 2000.0**3:.4e}", f"fsat = {R[name]['fsat']:.4f}", flush=True)
np.save(G + "pk_gal_c009.npy", R, allow_pickle=True)
