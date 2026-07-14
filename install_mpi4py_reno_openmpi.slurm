#!/bin/bash
#SBATCH --job-name=reno_mpi4py
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -p 4V100
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --output=install_mpi4py_reno_%j.log
#SBATCH --error=install_mpi4py_reno_%j.log

set -euo pipefail

echo "=== job ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PWD=$(pwd)"

echo "=== modules ==="
source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load openmpi/4.1.8-gnu-13.3.0
module list

echo "=== conda ==="
set +u
source /software/envs/anaconda3.env
conda activate reno
set -u
which python
python --version
python -m pip --version

echo "=== before ==="
python - <<'PY'
import importlib.util
for name in ("mpi4py", "cupy", "numpy"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else None}")
PY

echo "=== mpi compiler ==="
export MPICC="$(command -v mpicc)"
echo "MPICC=${MPICC}"
mpicc --showme:version
mpicc --showme:command
mpicc --showme:link

export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/mpi4py-build-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"
export PIP_NO_CACHE_DIR=1

echo "=== install mpi4py ==="
python -m pip install \
    --no-deps \
    --no-binary=mpi4py \
    --no-build-isolation \
    "mpi4py==3.1.6"

echo "=== after import ==="
python - <<'PY'
import importlib.util
import mpi4py
from mpi4py import MPI

print("mpi4py_version", mpi4py.__version__)
print("mpi4py_file", mpi4py.__file__)
print("mpi4py_MPI_so", importlib.util.find_spec("mpi4py.MPI").origin)
print("MPI_vendor", MPI.get_vendor())
print("MPI_library_version", MPI.Get_library_version().splitlines()[0])
PY

echo "=== linked libraries ==="
MPI_SO="$(python - <<'PY'
import importlib.util
print(importlib.util.find_spec("mpi4py.MPI").origin)
PY
)"
ldd "${MPI_SO}"

echo "=== 2-rank smoke test ==="
mpiexec -n 2 python -c 'from mpi4py import MPI; import socket; comm = MPI.COMM_WORLD; print("rank=%d size=%d host=%s vendor=%s lib=%s" % (comm.rank, comm.size, socket.gethostname(), MPI.get_vendor(), MPI.Get_library_version().splitlines()[0]), flush=True)'

echo "=== done ==="
date
