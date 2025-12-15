#!/bin/bash --login
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=rtx2080ti # or gtx1080ti / v100 depending on availability
#SBATCH --partition=batch 
#SBATCH --job-name=coral_app
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j-slurm.out
#SBATCH --error=log/%x-%j-slurm.err
# setup the environment

export ENV_PREFIX=$PWD/env
source /ibex/scratch/$USER/miniconda3/bin/activate
export CONDA_PKGS_DIRS=/ibex/user/${USER}/conda_cache
conda activate "$ENV_PREFIX"


# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "
this is the port from SLURM ${SLURM_STEP_RESV_PORTS}
To connect to the compute node ${COMPUTE_NODE} on Ibex running your Code Server, 
you need to create an ssh tunnel from your local machine to login node on Ibex 
using the following command.

ssh -L localhost:${SERVER_PORT}:${COMPUTE_NODE}:${SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${SERVER_PORT}
" 

# launch code server
cd app; 
streamlit run streamlit_starting_page.py --server.port=${SERVER_PORT}  
