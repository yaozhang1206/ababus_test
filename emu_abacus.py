"""emu_worker.py for custom boxes: patches lh_common.cosmologies() to {cid: params} for every cid, then runs it.

    python emu_abacus.py --cids 9009 9010 --params Om Ob h ns s8 -- --slab S --shm SHM --cat-dir DIR
Env: conda nn_test (as emu_worker.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np        # noqa: E402

import lh_common as C     # noqa: E402

i = sys.argv.index("--")
own, rest = sys.argv[1:i], sys.argv[i + 1:]
j = own.index("--params")
par = np.array([float(x) for x in own[j + 1:j + 6]])
cids = [int(x) for x in own[own.index("--cids") + 1:j]]
C.cosmologies = lambda: {c: par for c in cids}
sys.argv = [sys.argv[0]] + rest

import emu_worker         # noqa: E402

emu_worker.main()
