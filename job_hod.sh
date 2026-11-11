#!/bin/bash
# HOD galaxies on c009 halos (emulator / Abacus x 2 velocity copies) x (neutral, ext HOD), then P0/P2.
set -u
cd /u/yzhang116/test_discodj/lh4000/abacus
hostname; date
module purge
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for c in emu abacus_nbodyconc abacus_custom; do for h in neutral ext; do
  /u/yzhang116/.conda/envs/cmass/bin/python -u hod_c009.py --cat $c --hod $h > logs/hod_${c}_${h}.log 2>&1 &
done; done
wait
tail -n 2 logs/hod_*.log; date
export OMP_NUM_THREADS=64 PYTHONPATH=/u/yzhang116/test_discodj/lh4000
/u/yzhang116/.conda/envs/pm/bin/python -u pk_gal_c009.py
date
