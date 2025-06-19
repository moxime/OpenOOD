#! env bash

if [ -z $1 ]
then
    cd configs/postprocessors/
    ls *.yml
else
    method="$1"
    shift
    python main.py "$@" --config configs/datasets/cifar10/cifar10_raw.yml configs/networks/cvae_encoder.yml configs/datasets/cifar10/cifar10_ood.yml configs/preprocessors/base_preprocessor.yml  configs/postprocessors/$method.yml configs/pipelines/test/test_ood.yml --seed 0
fi
