# Environment for jobs on the cluster; sourced by the *.sbatch scripts.
# Edit the two variables below once, after installing (see README, "Running on the DIFA cluster").

# Path of the virtual environment created at installation:
RBDSMC_VENV="${RBDSMC_VENV:-$HOME/venvs/rbdsmc}"
# Modules to load before activating it (leave empty if the system python3 is >= 3.10):
RBDSMC_MODULES="${RBDSMC_MODULES:-}"

if [ -n "$RBDSMC_MODULES" ] && command -v module >/dev/null 2>&1; then
    module load $RBDSMC_MODULES
fi
# shellcheck disable=SC1091
source "$RBDSMC_VENV/bin/activate"

# The code is serial: stop NumPy/BLAS from starting one thread per core of the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
mkdir -p logs
