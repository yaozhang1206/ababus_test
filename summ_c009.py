"""P(k) and B(k) of c009 tracers exactly as ltu-cmass cmass/diagnostics/summ.py::summarize_tracer (box mode).

Calls summ.py's own functions, so the estimators/binning are identical:
  P(k): run_pypower_box -- pypower CatalogMesh, TSC, interlacing 2, mesh get_mesh_resolution(L, high_res=True)
        (= 2 * 128 per Gpc/h -> 512^3 for L = 2000), ells 0/2/4, k edges K_MIN = 0.01, DK_PK = 0.01, no shot-noise
        subtraction; real space ('Pk') and redshift space along z ('zPk', get_redshift_space_pos, axis = 2).
  B(k): run_bispectrum(backend='polybin') -- PolyBin3D Bk_ideal(discreteness_correction=True), lmax 2, TSC mesh
        get_mesh_resolution(L, high_res=False) (= 256^3), side bins 0.01..0.40 by 0.04; real ('Bk','Qk','bPk')
        and redshift space ('zBk','zQk','zbPk').
  Halos: optional summ.py-style density cut (keep the top n*L^3 by mass) and mass histogram (12.5..16, 100 edges).
  In-voxel uniform noise (config.bias.hod.noise_uniform) is off by default: --noise-cell L/N_nbody turns it on.

    python summ_c009.py --cmass /path/to/ltu-cmass --out-dir DIR \
        --tracer NAME KIND PATH [--tracer ...] [--density 3e-4] [--threads 32]
  KIND: gal  (hod_c009.py npy: x y z vx vy vz is_sat)
        emu  (DISCO-DJ + transformer catalog npy: x y z M vx vy vz conc)
        abacus (maho3 halos.h5, group '0.666667', log10 mass)
  With no --tracer, runs the six HOD galaxy catalogs of hod_c009.py.
Output: DIR/{NAME}.h5 with summ.py's keys (+ attrs Ngalaxies, nbar, boxsize, ...).
Env: ltu-cmass env (pypower, PolyBin3D, Pylians, hydra/omegaconf for importing summ.py).
"""
import argparse, os, sys, time
import numpy as np

L, Z = 2000.0, 0.5
OM, OB, H, NS, S8 = 0.3137721, 0.0493017, 0.6736, 0.9649, 0.811362     # AbacusSummit c009
GAL = "/work/hdd/bdne/yzhang116/abacus_c009/galaxies/"
DEFAULT = [(f"gal_{c}_{h}", "gal", GAL + f"gal_{c}_{h}.npy")
           for h in ("neutral", "ext") for c in ("emu", "abacus_nbodyconc", "abacus_custom")]


def load(kind, path):
    if kind == "gal":
        g = np.load(path)
        return g[:, :3].astype(np.float32), g[:, 3:6].astype(np.float32), None
    if kind == "emu":
        c = np.load(path)
        return c[:, :3].astype(np.float32), c[:, 4:7].astype(np.float32), np.log10(c[:, 3]).astype(np.float32)
    if kind == "abacus":
        import h5py
        with h5py.File(path) as f:
            g = f["0.666667"]
            return (g["pos"][:].astype(np.float32) % L, g["vel"][:].astype(np.float32), g["mass"][:].astype(np.float32))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmass", default="/u/yzhang116/ltu-cmass", help="ltu-cmass repo root (contains cmass/)")
    ap.add_argument("--out-dir", default=GAL + "summ/")
    ap.add_argument("--tracer", nargs=3, action="append", metavar=("NAME", "KIND", "PATH"))
    ap.add_argument("--density", type=float, default=None, help="summ.py density cut (h/Mpc)^3, halos only")
    ap.add_argument("--noise-cell", type=float, default=0.0, help="uniform in-voxel noise width (Mpc/h), 0 = off")
    ap.add_argument("--no-bk", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    a = ap.parse_args()
    sys.path.insert(0, a.cmass)
    from astropy.cosmology import FlatLambdaCDM
    from cmass.diagnostics.summ import run_pypower_box, run_bispectrum
    from cmass.diagnostics.tools import get_mesh_resolution
    from cmass.diagnostics.calculations import MA, MAz, get_redshift_space_pos
    import h5py
    cosmo = FlatLambdaCDM(H0=100 * H, Om0=OM, Ob0=OB)
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    for name, kind, path in (a.tracer or DEFAULT):
        if not os.path.exists(path):
            print(f"skip {name}: {path} missing", flush=True)
            continue
        t0 = time.time()
        pos, vel, mass = load(kind, path)
        if a.density is not None and mass is not None:          # summ.py: keep the top Ncut tracers by mass
            ncut = int(a.density * L ** 3)
            keep = np.argsort(mass)[-ncut:]
            pos, vel, mass = pos[keep], vel[keep], mass[keep]
        attrs = dict(Ngalaxies=len(pos), boxsize=L, nbar=len(pos) / L ** 3,
                     log10nbar=np.log10(len(pos)) - 3 * np.log10(L), high_res=True, redshift=Z, source=path,
                     Om=OM, Ob=OB, h=H, ns=NS, sigma8=S8, noise_cell=a.noise_cell)
        if a.noise_cell > 0:
            pos = pos + rng.uniform(-a.noise_cell / 2, a.noise_cell / 2, size=pos.shape).astype(np.float32)
        pos = np.ascontiguousarray(pos % L, dtype=np.float32)
        out = {}
        N, MAS = get_mesh_resolution(L, high_res=True, use_ngp=False)
        out.update(run_pypower_box(pos, L, N, axis=2, MAS=MAS, num_threads=a.threads, use_rsd=False))
        zpos = get_redshift_space_pos(pos.copy(), vel.copy(), L, cosmo, Z, axis=2)
        out.update(run_pypower_box(zpos, L, N, axis=2, MAS=MAS, num_threads=a.threads, use_rsd=True))
        print(f"{name}: N = {len(pos)}, nbar = {attrs['nbar']:.4e}, Pk mesh {N} {MAS}, {time.time() - t0:.0f} s", flush=True)
        if not a.no_bk:
            N, MAS = get_mesh_resolution(L, high_res=False, use_ngp=False)
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out.update(run_bispectrum(field, L, axis=2, MAS=MAS, num_threads=a.threads, use_rsd=False, backend="polybin"))
            field = MAz(pos, vel, L, N, cosmo, Z, MAS=MAS, axis=2).astype(np.float32)
            out.update(run_bispectrum(field, L, axis=2, MAS=MAS, num_threads=a.threads, use_rsd=True, backend="polybin"))
            del field
            print(f"{name}: Bk mesh {N} {MAS}, {time.time() - t0:.0f} s", flush=True)
        if mass is not None:
            be = np.linspace(12.5, 16, 100)
            out["mass_bins"], out["mass_hist"] = be, np.histogram(mass, bins=be)[0]
        with h5py.File(os.path.join(a.out_dir, f"{name}.h5"), "w") as f:
            for k, v in out.items():
                f.create_dataset(k, data=np.asarray(v))
            for k, v in attrs.items():
                f.attrs[k] = v


if __name__ == "__main__":
    main()
