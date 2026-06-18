python -m openood.utils.config --config \
       configs/pipelines/test/test_tta_ood.yml \
       configs/networks/resnet18_32x32.yml \
       configs/datasets/cifar10/cifar10_tta_ood.yml \
       configs/datasets/cifar10/cifar10.yml \
       "$@"
