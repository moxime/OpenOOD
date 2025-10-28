#!/bin/bash
#SBATCH --output=/mnt/beegfs/home/ossonce/tia-dl-ossonce/ooddict/jobs/job-%j.out
#SBATCH --error=/mnt/beegfs/home/ossonce/tia-dl-ossonce/ooddict/jobs/job-%j.err
#SBATCH --mail-user=mossonce@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/mnt/beegfs/home/ossonce/tia-dl-ossonce/openood
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=v100

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


seed=0

# id / network: $1
case $1 in

    cifar10)
	idset=cifar10
	network=resnet18_32x32
	ckpt_suffix=base_e100_lr0.1_default
    ;;

    cifar100)
	idset=cifar100
	network=resnet18_32x32
	ckpt_suffix=base_e100_lr0.1_default
    ;;

    imagenet)
	idset=imagenet
	network=resnet50
	ckpt="checkpoints/resnet50-0676ba61.pth"

    ;;

    imagenet200)
	idset=imagenet200
	network=resnet18_32x32
	ckpt_suffix=base_e90_lr0.1_default
    ;;

esac

if [ -z $ckpt ] ; then
    ckpt=checkpoints/"$idset"_"network"_"$ckpt_suffix"/s"$seed"/best.ckpt
fi

method=$2

ls configs/datasets/$idset/$idset.yml \
       configs/networks/$network.yml \
       configs/datasets/$idset/"$idset"_ood.yml \
       configs/preprocessors/base_preprocessor.yml \
       configs/postprocessors/$method.yml \
       configs/pipelines/test/test_ood.yml \
       $ckpt

echo python main.py --config configs/datasets/$idset/$idset.yml \
       configs/networks/$network.yml \
       configs/datasets/$idset/"$idset"_ood.yml \
       configs/preprocessors/base_preprocessor.yml \
       configs/postprocessors/$method.yml \
       configs/pipelines/test/test_ood.yml \
       --seed $seed \
       --mark $SLURM_JOB_ID


