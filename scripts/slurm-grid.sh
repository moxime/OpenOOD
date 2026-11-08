#!/bin/bash

beta_=(0.1 0.2 0.5 1 2 5 10)
pad_=(patched_id)
epochs_=(100 200 400 600 800 1000)


while :; do

    case $1 in

	-B)
	    shift
	    beta_=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                beta_+=("$1")
                shift
	    done	    
	;;

	-E)
	    shift
	    epochs_=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                epochs_+=("$1")
                shift
	    done	    
	;;

	-P)
	    shift
	    pad_=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                pad_+=("$1")
                shift
	    done	    
	;;
	
	--dry-run )
	    pre_cmd=echo
	    shift
	;;

	* )
	    break
	    ;;
    esac
done

slurm_script=$(dirname $0)/slurm-job.sh


for beta in ${beta_[@]}
do
    for epochs in ${epochs_[@]}
    do
	for pad in ${pad_[@]}
	do
	    $pre_cmd sbatch $slurm_script $@ \
		     --postprocessor.ft.epochs $epochs \
		     --postprocessor.ft.beta $beta \
		     --ood_dataset.padding.datasets $pad
	done
    done
done
