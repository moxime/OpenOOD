#!/bin/bash

beta_=(0.1 0.2 0.5 1 2 10)
pad_=(lsunr patched_id)
epochs_=(100 200 400 600 800 1000)

while :; do

    case $1 in

	--beta )
	;;
	* )
	    break
	    ;;
    esac
    shift
done

slurm_script=$(dirname $0)/slurm-job.sh


for beta in ${beta_[@]}
do
    for epochs in ${epochs_[@]}
    do
	for pad in ${pad_[@]}
	do
	    echo sbatch $slrum_script $@ \
		 --postprocessor.ft.epochs $epochs \
		 --postprocessor.ft.beta $beta \
		 --ood_dataset.padding.datasets $pad
	done
    done
done
