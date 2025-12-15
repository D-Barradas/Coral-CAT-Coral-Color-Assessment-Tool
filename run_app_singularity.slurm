#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=rtx2080ti # or gtx1080ti / v100 depending on availability
#SBATCH --partition=batch
#SBATCH --job-name=coral_app
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j-slurm.out
#SBATCH --error=log/%x-%j-slurm.err

set -euo pipefail

### Load the modules you need for your job
# module purge
# module load rl9-gpustack
module load singularity

### Helper Functions Definitions
checking_container_image() {
    ## 1. Define variables

    ## 2. Validate and pull the image if it is not available or is corrupted
    # Check if image already exists
    if [[ ! -f "${SIF_FILE_PATH}" ]]; then
        echo "[FAIL] Image not found. Pulling from ${IMAGE_NAME} ..."
        singularity pull "${SIF_FILE_PATH}" "docker://${IMAGE_NAME}"
    else
        echo "[OK] Found existing SIF file: ${SIF_FILE_PATH}"

        # Try running singularity inspect (safe way to check validity)
        if singularity inspect "${SIF_FILE_PATH}" > /dev/null 2>&1; then
            echo "[OK] Validation successful. Image is usable."
        else
            echo "[FAIL] Existing file is corrupted or invalid. Re-pulling..."
            rm -f "${SIF_FILE_PATH}"
            singularity pull "${SIF_FILE_PATH}" "docker://$IMAGE_NAME"
        fi
    fi
    echo "[OK] Using image: ${SIF_FILE_PATH}"
}


## START OF THE SCRIPT
### Define container and package paths
echo "=== Checking Container Image ==="

IMAGE_NAME="dxbarradas/coralcat_v1.0:amd64"
SIF_FILE_NAME="coralcat_v1.0_amd64.sif"
SIF_FILE_PATH="/ibex/user/${USER}/containers/${SIF_FILE_NAME}"

echo "=== Preparing Container Environment ==="
checking_container_image

# Create containers directory if needed
mkdir -p /ibex/user/${USER}/containers


echo "=== Setup SSH tunneling information  ==="

node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

instructions="

============================================================
CoralCAT is starting on compute node: ${node}
============================================================

To connect from your local machine, run this in a NEW terminal:

    ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} ${user}@glogin.ibex.kaust.edu.sa

Then copy the URL shown like http://localhost:${port}
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

# Run CoralCat inside the Singularity container
# Note: All options (--nv, -B, --env) must come BEFORE the container path

singularity exec --nv \
    "${SIF_FILE_PATH}" \
    /bin/bash -lc "echo -e '${instructions}'; cd app ; 
    streamlit run streamlit_starting_page.py --server.port=${port}"
    
    