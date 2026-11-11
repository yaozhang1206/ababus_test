"""Producer for custom boxes (e.g. AbacusSummit c009 ph000 phases): DISCO-DJ 2 Gpc/h + transformer input fields.

Same N-body / field code as the LH4000 production (run_2gpc.run_box), but each IC is a precomputed z = 0 linear
delta(x) file (build_ic.py, or ic_worker.py output) and the cosmology is given on the command line.  Writes
SHM/cids.txt, SHM/{cid}/slab{d}.npy, SHM/producer.done for emu_abacus.py workers; before box n+1 it waits for
the workers' gpu_free flags of box n (as run_2gpc).

    python run_abacus.py --cids 9009 9010 --ics IC1.npy IC2.npy --params Om Ob h ns s8 --shm SHM
Env: as run_2gpc.py (conda pm + workspace DISCO-DJ/src on PYTHONPATH).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np        # noqa: E402

import lh_common as C     # noqa: E402
import run_2gpc as R      # noqa: E402  (sets the DISCO_* env and builds the 4-GPU meshes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", type=int, nargs="+", required=True)
    ap.add_argument("--ics", nargs="+", required=True)
    ap.add_argument("--params", type=float, nargs=5, required=True, metavar=("Om", "Ob", "h", "ns", "s8"))
    ap.add_argument("--shm", required=True)
    a = ap.parse_args()
    assert len(a.cids) == len(a.ics)
    with open(os.path.join(a.shm, "cids.txt.tmp"), "w") as fh:
        fh.write("\n".join(map(str, a.cids)) + "\n")
    os.replace(os.path.join(a.shm, "cids.txt.tmp"), os.path.join(a.shm, "cids.txt"))
    prev = None
    try:
        for cid, ic in zip(a.cids, a.ics):
            if prev is not None:
                C.wait_for(lambda: all(os.path.exists(C.gpu_free_flag(a.shm, prev, s)) for s in range(C.NSLAB)),
                           3600, what=f"gpu_free for {prev}")
            t0 = time.perf_counter()
            delta = np.load(ic)
            R.log(f"IC {ic}: std {delta.std():.4f}, load {time.perf_counter() - t0:.0f} s; halo {R.M.halo_size}")
            t = R.run_box(cid, np.asarray(a.params), delta, a.shm)
            del delta
            R.log(f"BOX {cid} params {a.params} " + json.dumps({k: round(v, 2) for k, v in t.items()}))
            prev = cid
        open(os.path.join(a.shm, "producer.done"), "w").close()
    except BaseException:
        open(os.path.join(a.shm, "producer.failed"), "w").close()
        raise


if __name__ == "__main__":
    main()
