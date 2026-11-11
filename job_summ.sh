#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=summ_c009
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# summ.py-style Pk/Bk (pypower + PolyBin3D) of the c009 HOD galaxies and, at the same density cut, the halos.
# Set PY (python of an env with pypower, PolyBin3D, Pylians, hydra/omegaconf), CMASS (ltu-cmass root) and the paths.
set -u
PY=${PY:-python}
CMASS=${CMASS:-/u/yzhang116/ltu-cmass}
HERE=${HERE:-/u/yzhang116/test_discodj/lh4000/abacus}
OUT=${OUT:-/work/hdd/bdne/yzhang116/abacus_c009/galaxies/summ}
CAT=${CAT:-/work/hdd/bdne/yzhang116/abacus_c009/catalogs}
AB=${AB:-/work/hdd/bdne/maho3/cmass-ili/abacus}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
hostname; date
# 1) HOD galaxies (hod_c009.py outputs)
$PY -u $HERE/summ_c009.py --cmass $CMASS --out-dir $OUT
# 2) halos with summ.py's density cut (top n * L^3 by mass), n = 3e-4
$PY -u $HERE/summ_c009.py --cmass $CMASS --out-dir $OUT --density 3e-4 \
   --tracer halo_emu emu $CAT/catalog_2gpc_9009.npy \
   --tracer halo_abacus_custom abacus $AB/custom/L2000-N256/30/halos.h5 \
   --tracer halo_abacus_nbodyconc abacus $AB/nbodyconc/L2000-N256/30/halos.h5
$PY $HERE/plot_summ_c009.py --dir $OUT
$PY $HERE/plot_summ_c009.py --dir $OUT --pairs halo_emu:halo_abacus_custom halo_emu:halo_abacus_nbodyconc --out $HERE/summ_c009_halo_cmp.png
date
