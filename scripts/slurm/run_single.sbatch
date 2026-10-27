#!/bin/bash
# One simulation on the DIFA OPH cluster (Slurm).  Submit from the repository root:
#     sbatch scripts/slurm/run_single.sbatch configs/test_simplest.yaml --set numerics.n_test=50000
# Everything after the script name is passed to scripts/run_single.py.
#
# Adjust the #SBATCH lines to your needs; lines starting with "##SBATCH" are inactive.
#SBATCH --job-name=rbdsmc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1        # the code is serial (NumPy); one core per run
#SBATCH --mem=2G                 # 1e5 test particles need < 1 GB
#SBATCH --time=02:00:00          # 1e4 particles x 200 ms ~ 2-4 min; 1e5 ~ 20-40 min
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --qos=debug             # max 15 min, no billing: use for a first test
##SBATCH --qos=long              # up to 72 h, low priority
##SBATCH --constraint=blade      # or matrix: ask your sector contact which to use
##SBATCH --account=oph           # only if your project requires it
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=name.surname00@unibo.it

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/slurm/env.sh

echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date)"
python scripts/run_single.py "$@" --quiet
echo "end $(date)"
