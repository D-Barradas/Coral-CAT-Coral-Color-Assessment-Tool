#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --job-name=build_coral_app
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
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
# Create containers directory if needed
mkdir -p /ibex/user/${USER}/containers

checking_container_image

