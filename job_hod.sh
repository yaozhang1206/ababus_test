#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=hod_c009
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/u/yzhang116/test_discodj/lh4000/abacus/logs/%x_%j.out
# HOD galaxies on c009 halos (emulator / Abacus x 2 velocity copies) x (neutral, ext HOD), then P0/P2.
set -u
cd /u/yzhang116/test_discodj/lh4000/abacus
hostname; date
module purge; module load cudatoolkit/25.5_12.9   # halotools' cython kernels in env cmass link libcudart.so.12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for c in emu abacus_nbodyconc abacus_custom; do for h in neutral ext; do
  /u/yzhang116/.conda/envs/cmass/bin/python -u hod_c009.py --cat $c --hod $h > logs/hod_${c}_${h}.log 2>&1 &
done; done
wait
tail -n 2 logs/hod_*.log; date
module purge
export OMP_NUM_THREADS=64 PYTHONPATH=/u/yzhang116/test_discodj/lh4000
/u/yzhang116/.conda/envs/pm/bin/python -u pk_gal_c009.py
date
