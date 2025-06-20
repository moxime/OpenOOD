#! env bash

methods=("msp" "odin" "mds"  "ebo" "gradnorm" "vim" "react" "ash") 

for method in "${methods[@]}"; do
    python main.py "$@" --config configs/datasets/cifar10/cifar10_raw.yml configs/networks/cvae_encoder.yml configs/datasets/cifar10/cifar10_ood.yml configs/preprocessors/base_preprocessor.yml  configs/postprocessors/$method.yml configs/pipelines/test/test_ood.yml --seed 0
done
