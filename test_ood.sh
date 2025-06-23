#! env bash

methods=("msp" "odin" "mds"  "ebo" "gradnorm" "vim" "react" "ash") 
gammas=(0)
while [[ $# -gt 0 ]]; do
    case $1 in

	-m)
	    methods=()
	    shift
	    while [[ $# -gt 0 ]]; do
		case $1 in
		    -*)
			break
			;;
		    *)
			methods+=("$1")
			shift
			;;
		esac
	    done
	    ;;
	-g)
	    gammas=()
	    shift
	    while [[ $# -gt 0 ]]; do
		case $1 in
		    -*)
			break
			;;
		    *)
			gammas+=("$1")
			shift
			;;
		esac
	    done
	    ;;
	*)
	    break
	    ;;
    esac
done


	  
for method in "${methods[@]}"; do
    for gamma in "${gammas[@]}"; do
	python main.py "$@" --config configs/datasets/cifar10/cifar10_raw.yml configs/networks/cvae_encoder.yml configs/datasets/cifar10/cifar10_ood.yml configs/preprocessors/base_preprocessor.yml  configs/postprocessors/$method.yml configs/pipelines/test/test_ood.yml --seed 0 --network.gamma "$gamma"
    done
done
