#!/bin/bash
# Parameter scan as a Slurm job array: one array task per line of a scan list
# (written by scripts/make_scan_list.py).  Submit from the repository root:
#
#     N=$(grep -cv '^\s*\(#\|$\)' configs/scans/power_height.txt)
#     sbatch --array=1-$N%20 scripts/slurm/run_array.sbatch configs/scans/power_height.txt
#
# "%20" limits the number of tasks running at the same time (be fair to other users).
# Finished runs (summary.json present) are skipped, so the same command can be resubmitted
# after a failure or a timeout.
#SBATCH --job-name=rbdsmc_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --qos=long
##SBATCH --constraint=blade
##SBATCH --account=oph
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=name.surname00@unibo.it

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/slurm/env.sh

LIST="$1"; shift || true
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-?}  start $(date)"
python scripts/run_list.py "$LIST" --line "${SLURM_ARRAY_TASK_ID:?run as an array job}" "$@"
echo "end $(date)"
