"""10-parameter HOD (ltu-cmass Zheng07 + assembly bias + velocity/concentration bias, as in
NN/inference/statistics/gal_pk_bk_wide_hod_ext.py) on 2 Gpc/h c009 halo catalogs, tuned to n_gal = 3e-4.

Per catalog, logMmin, logM0 and logM1 are shifted by a common Delta found by bisection on the expected
<N_cen> + <N_sat> (assembly bias only redistributes galaxies at fixed mass).  Then halotools populate_mock.
    python hod_c009.py --cat emu|abacus_nbodyconc|abacus_custom --hod neutral|ext [--seed 0]
Env: conda cmass (halotools).  Output: galaxies/gal_{cat}_{hod}.npy  float32 [x y z vx vy vz is_sat], + .json
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq
sys.path.insert(0, "/u/yzhang116/ltu-cmass")
from cmass.bias.tools.hod_models import Zheng07                      # noqa: E402
from astropy.cosmology import FlatLambdaCDM                          # noqa: E402
from halotools.sim_manager import UserSuppliedHaloCatalog            # noqa: E402
from halotools.empirical_models import halo_mass_to_halo_radius, NFWProfile  # noqa: E402

L, Z, NT = 2000.0, 0.5, 3e-4
OM, OB, H = 0.3137721, 0.0493017, 0.6736
OUT = "/work/hdd/bdne/yzhang116/abacus_c009/galaxies/"
AB = "/work/hdd/bdne/maho3/cmass-ili/abacus/{}/L2000-N256/30/halos.h5"
BASE = dict(logMmin=13.03, sigma_logM=0.38, logM0=13.27, logM1=14.08, alpha=0.76)   # Reid+2014 CMASS
EXT = {"neutral": dict(abias_cen=0.0, abias_sat=0.0, eta_cen=0.0, eta_sat=1.0, eta_conc=1.0),
       "ext": dict(abias_cen=0.3, abias_sat=-0.2, eta_cen=0.2, eta_sat=1.3, eta_conc=0.8)}


def load(cat):
    if cat == "emu":
        c = np.load("/work/hdd/bdne/yzhang116/abacus_c009/catalogs/catalog_2gpc_9009.npy").astype(np.float64)
        return c[:, :3], c[:, 4:7], c[:, 3], c[:, 7]
    import h5py
    with h5py.File(AB.format(cat.split("_")[1])) as f:
        g = f["0.666667"]
        pos, vel, m = g["pos"][:].astype(np.float64), g["vel"][:].astype(np.float64), 10.0 ** g["mass"][:].astype(np.float64)
        conc = g["concentration"][:].astype(np.float64) if "concentration" in g else None
    return pos % L, vel, m, conc


def expected_n(lm, p):
    nc = 0.5 * (1 + erf((lm - p["logMmin"]) / p["sigma_logM"]))
    ns = nc * (np.clip(10 ** lm - 10 ** p["logM0"], 0, None) / 10 ** p["logM1"]) ** p["alpha"]
    return nc.sum() / L ** 3, ns.sum() / L ** 3


def shifted(d):
    return {k: (v + d if k in ("logMmin", "logM0", "logM1") else v) for k, v in BASE.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True); ap.add_argument("--hod", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    pos, vel, m, conc = load(a.cat)
    lm = np.log10(m)
    d = brentq(lambda d: sum(expected_n(lm, shifted(d))) - NT, -1, 1, xtol=1e-6)
    p5 = shifted(d); nc, ns = expected_n(lm, p5)
    e = EXT[a.hod]
    theta = dict(p5, mean_occupation_centrals_assembias_param1=e["abias_cen"],
                 mean_occupation_satellites_assembias_param1=e["abias_sat"],
                 eta_vb_centrals=e["eta_cen"], eta_vb_satellites=e["eta_sat"], conc_gal_bias_satellites=e["eta_conc"])
    print(f"{a.cat} {a.hod}: {len(m)} halos, Delta = {d:+.4f}, expected n = {(nc + ns):.4e}, fsat = {ns / (nc + ns):.4f}", flush=True)

    cosmo = FlatLambdaCDM(H0=100 * H, Om0=OM, Ob0=OB)
    mdef = "200c"
    if conc is None:
        conc = NFWProfile(cosmology=cosmo, conc_mass_model="dutton_maccio14", mdef=mdef, redshift=Z).conc_NFWmodel(prim_haloprop=m)
    n = len(m)
    halocat = UserSuppliedHaloCatalog(
        halo_x=pos[:, 0], halo_y=pos[:, 1], halo_z=pos[:, 2], halo_vx=vel[:, 0], halo_vy=vel[:, 1], halo_vz=vel[:, 2],
        halo_nfw_conc=conc, halo_redshift=np.full(n, Z), halo_id=np.arange(n), halo_hostid=np.zeros(n, dtype=int),
        halo_upid=np.full(n, -1.0), halo_local_id=np.arange(n, dtype="i8"), cosmology=cosmo, redshift=Z, particle_mass=1,
        Lbox=np.array([L, L, L]), mdef=mdef, halo_m200c=m, halo_r200c=halo_mass_to_halo_radius(m, cosmo, Z, mdef),
        halo_mvir=np.full(n, np.nan), halo_rvir=np.full(n, np.nan))
    hm = Zheng07(mass_def=mdef, assem_bias=True, vel_assem_bias=True)
    hm.set_parameters(theta); hm.set_occupation(); hm.set_profiles(cosmology=cosmo, zf=Z)
    model = hm.get_model()
    model.populate_mock(halocat, seed=a.seed, halo_mass_column_key=f"halo_m{mdef}")
    g = model.mock.galaxy_table
    sat = (np.array(g["gal_type"]) == "satellites")
    out = np.column_stack([np.array([g["x"], g["y"], g["z"]]).T % L, np.array([g["vx"], g["vy"], g["vz"]]).T, sat]).astype(np.float32)
    np.save(OUT + f"gal_{a.cat}_{a.hod}.npy", out)
    summ = dict(cat=a.cat, hod=a.hod, delta=d, theta=theta, n_expected=nc + ns, fsat_expected=ns / (nc + ns),
                ngal=len(out), n=len(out) / L ** 3, fsat=float(sat.mean()), seconds=time.time() - t0)
    json.dump(summ, open(OUT + f"gal_{a.cat}_{a.hod}.json", "w"), indent=1)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ.items() if k != "theta"}), flush=True)


if __name__ == "__main__":
    main()
