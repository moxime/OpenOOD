#!/bin/bash
#SBATCH --output=/mnt/beegfs/home/ossonce/tia-dl-ossonce/ooddict/jobs/job-%j.out
#SBATCH --error=/mnt/beegfs/home/ossonce/tia-dl-ossonce/ooddict/jobs/job-%j.err
#SBATCH --mail-user=mossonce@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/mnt/beegfs/home/ossonce/tia-dl-ossonce/ooddict
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=v100
# export LD_LIBRARY_PATH=~/miniconda3/envs/test-cuda
echo $SLURM_JOB_ID is started
source ~/miniconda3/etc/profile.d/conda.sh

CONDA_ENV="openood"

conda activate $CONDA_ENV 

export CUDA_LAUNCH_BLOCKING=1


echo "started job by $(whoami)"

nvidia-smi
nvidia-smi -L

ulimit -n 4096
echo "setting ulimit -n to $(ulimit -n)"

. ./bash.sh
