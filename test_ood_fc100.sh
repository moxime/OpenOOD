#! env bash

if [ -z $1 ]
then
    cd configs/postprocessors/
    ls *.yml
else
    python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood_fc100.yml configs/preprocessors/base_preprocessor.yml configs/networks/vgg19_32x32.yml configs/postprocessors/$1.yml configs/pipelines/test/test_ood.yml --seed 0
fi
