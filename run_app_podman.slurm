#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=rtx2080ti # or gtx1080ti / v100 depending on availability
#SBATCH --partition=batch 
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=8G
#SBATCH --job-name=coral_app
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j-slurm.out
#SBATCH --error=log/%x-%j-slurm.err

set -euo pipefail

# setup the environment

export ROOT_DIR=/ibex/user/${USER}/podman_image
mkdir -p /run/user/${UID}/bus ${ROOT_DIR}


# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "
============================================================
CoralCAT is starting on compute node: ${node}
============================================================

To connect from your local machine, run this in a NEW terminal:

    ssh -L ${SERVER_PORT}:${COMPUTE_NODE}.ibex.kaust.edu.sa:${SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa

Then copy the URL shown like http://localhost:${SERVER_PORT}
and paste it into your browser on your local machine.

============================================================
IMPORTANT: Shutdown CoralCat 
============================================================

When finished, close your browser and go back to the terminal
where you submitted this job and type:

scancel ${SLURM_JOB_ID}

If you do not shutdown CoralCat, the job will continue running until the
time limit is reached.

============================================================
"

# prune old containers
podman --root=${ROOT_DIR} container prune -f > /dev/null 2>&1 || true

# searhc in images if the image exists otherwise pull it
if ! podman --root=${ROOT_DIR} image exists coralcat_v1.0:amd64; then
    echo "[FAIL] Image not found. Pulling from local build ..."
    podman --root=$ROOT_DIR pull docker.io/dxbarradas/coralcat_v1.0:amd64
    # podman --root=${ROOT_DIR} pull docker-daemon:coralcat_v1.0:amd64
else
    echo "[OK] Found existing Podman image: coralcat_v1.0:amd64"
fi


# Launch CoralCat inside Podman container
podman run --root=${ROOT_DIR} --rm -v ${PWD}:/workspace/ \
    --device=nvidia.com/gpu=all \
    --security-opt=label=disable \
    -p ${SERVER_PORT}:8501 \
    --name CoralCat \
    coralcat_v1.0:amd64 streamlit run streamlit_starting_page.py --server.port=8501 