#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=abacus_c009
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/u/yzhang116/test_discodj/lh4000/abacus/logs/%x_%j.out
# AbacusSummit_base_c009_ph000 at z = 0.5 with DISCO-DJ (2 Gpc/h, 1024^3, PM 2048^3) + transformer:
#   9009 = Abacus phases (build_ic.py), 9010 = same cosmology, random N-GenIC phases (seed 9010).
set -u
export DISCO_HALO_SIZE=${DISCO_HALO_SIZE:-240}
HERE=/u/yzhang116/test_discodj/lh4000
AB=$HERE/abacus
OUT=/work/hdd/bdne/yzhang116/abacus_c009
CAT_DIR=$OUT/catalogs
PARAMS="0.3137721 0.0493017 0.6736 0.9649 0.811362"
CIDS="9009 9010"
SHM=/dev/shm/abacus_${SLURM_JOB_ID}
mkdir -p $SHM $AB/logs $CAT_DIR; trap 'rm -rf $SHM /dev/shm/abacus_wn_full.npy' EXIT
LOG=$AB/logs/abacus_${SLURM_JOB_ID}
PY=/u/yzhang116/.conda/envs/pm/bin/python
hostname; nvidia-smi -L; date

# ---- ICs (CPU): Abacus phases, and random phases (compiled discodj_native: no DISCO-DJ/src on PYTHONPATH)
(
  module purge; module load gsl
  export LD_LIBRARY_PATH=/sw/user/gsl/lib:${LD_LIBRARY_PATH:-}   # discodj_native needs libgsl.so.28
  export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8 PYTHONPATH=$HERE
  [ -f $OUT/ic_c009_ph000_N1024.npy ] || $PY -u $AB/build_ic.py --out $OUT/ic_c009_ph000_N1024.npy > ${LOG}_ic.log 2>&1
  [ -f $OUT/ic_c009_random9010_N1024.npy ] || $PY -u -c "
import numpy as np, os, ic_dj
d = ic_dj.delta_x_host(9010, 1024, 2000.0, '$OUT/Pk_mm_z=0.000.txt', 0.811362, workers=32)
np.save('$OUT/ic_c009_random9010_N1024.tmp.npy', d); os.replace('$OUT/ic_c009_random9010_N1024.tmp.npy', '$OUT/ic_c009_random9010_N1024.npy')
print('random IC std', d.std(), flush=True)" >> ${LOG}_ic.log 2>&1
)
cat ${LOG}_ic.log; date
[ -f $OUT/ic_c009_ph000_N1024.npy ] && [ -f $OUT/ic_c009_random9010_N1024.npy ] || { echo "IC failed"; exit 1; }

# ---- producer: JAX / DISCO-DJ on all 4 GPUs (conda pm)
(
  module purge; module load cuda/12.4.0 cudnn/9.3.0.75 gsl
  export OFI_PLUGIN=none; source /u/yzhang116/test_discodj/bitcmp/delta_ofi_env.sh
  export PYTHONPATH=/u/yzhang116/test_discodj/DISCO-DJ/src:$HERE
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export DISCO_PM_FFT_BACKEND=JAX_RFFT OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8
  cd $HERE
  $PY -u $AB/run_abacus.py --cids $CIDS --ics $OUT/ic_c009_ph000_N1024.npy $OUT/ic_c009_random9010_N1024.npy \
     --params $PARAMS --shm $SHM > ${LOG}_producer.log 2>&1
  echo "producer exit $?" >> ${LOG}_producer.log
) &
PROD=$!

# ---- 4 emulator workers (conda nn_test), one per GPU
for s in 0 1 2 3; do
(
  module purge; module load cudatoolkit/25.5_12.9
  source ~/.bashrc >/dev/null 2>&1; conda activate nn_test
  export CC=gcc CXX=g++ CUDA_VISIBLE_DEVICES=$s OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=8
  export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}_$s
  cd $HERE
  python -u $AB/emu_abacus.py --cids $CIDS --params $PARAMS -- --slab $s --shm $SHM --cat-dir $CAT_DIR \
     > ${LOG}_worker$s.log 2>&1
  echo "worker $s exit $?" >> ${LOG}_worker$s.log
) &
done
wait $PROD
wait
grep -h "BOX\|exit\|Error\|Traceback" ${LOG}_producer.log | cut -c1-600
grep -h "CATALOG\|exit\|Error\|Traceback" ${LOG}_worker*.log | cut -c1-300
date

# ---- statistics vs Abacus
(
  module purge; export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8 PYTHONPATH=$HERE
  $PY -u $AB/compare_abacus.py --emu $CAT_DIR/catalog_2gpc_9009.npy --emu-random $CAT_DIR/catalog_2gpc_9010.npy \
     --out $OUT/stats_c009.npy --fig $AB/abacus_c009_cmp.png > ${LOG}_stats.log 2>&1
)
cat ${LOG}_stats.log
echo "DONE $(date)"
