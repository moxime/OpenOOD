#!/bin/bash
#SBATCH --output=/mnt/beegfs/home/ossonce/tia-dl-ossonce/openood/jobs/job-%A-%a.out
#SBATCH --error=/mnt/beegfs/home/ossonce/tia-dl-ossonce/openood/jobs/job-%A-%a.err
#SBATCH --mail-user=mossonce@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/mnt/beegfs/home/ossonce/tia-dl-ossonce/openood
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --constraint=v100
#SBATCH --array=0-10


dataset=cifar10

while :; do
    case $1 in
	-d )
	    shift
	    dataset=$1
	    ;;
	* )
	    break
	    ;;
    esac
    shift
done



echo $SLURM_JOB_ID is started
source ~/miniconda3/etc/profile.d/conda.sh

CONDA_ENV="openood"

conda activate $CONDA_ENV 

export CUDA_LAUNCH_BLOCKING=1

echo "started job by $(whoami)"

# nvidia-smi
# nvidia-smi -L

ulimit -n 4096
echo "setting ulimit -n to $(ulimit -n)"


seed=0

unset ckpt

# id / network: $1
case $dataset in

    cifar10)
    # network=resnet18_32x32
    # ckpt_suffix=base_e100_lr0.1_default
	network=resnet20
	ckpt="checkpoints/resnet20/best.ckpt"
    ;;

    cifar100)
	network=resnet18_32x32
	ckpt_suffix=base_e100_lr0.1_default
    ;;

    imagenet)
	network=resnet50
	ckpt="checkpoints/resnet50-0676ba61.pth"

    ;;

    imagenet200)
	network=resnet18_224x224
	ckpt_suffix=base_e90_lr0.1_default
    ;;

esac

if [ -z $ckpt ] ; then
    ckpt=checkpoints/"$dataset"_"$network"_"$ckpt_suffix"/s"$seed"/best.ckpt
fi


ls -1 configs/datasets/$dataset/$dataset.yml \
   configs/networks/$network.yml \
   configs/datasets/$dataset/"$dataset"_tta_ood.yml \
   configs/preprocessors/base_preprocessor.yml \
   configs/postprocessors/$method.yml \
   configs/pipelines/test/test_tta_ood.yml \
   $ckpt

python main.py --config configs/datasets/$dataset/$dataset.yml \
       configs/networks/$network.yml \
       configs/datasets/$dataset/"$dataset"_tta__ood.yml \
       configs/preprocessors/base_preprocessor.yml \
       configs/postprocessors/$method.yml \
       configs/pipelines/test/test_tta_ood.yml \
       --network.checkpoint $ckpt \
       --seed $seed \
       --mark $SLURM_JOB_ID \
       "$@"
