import os
import functools
import numpy as np
import torch
from numpy import load
from torch.utils.data import DataLoader, RandomSampler, Subset

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .mixture_dataset import IDOODDataset, IDOODSampler, MixtureDataset, MixtureTimeSampler
from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .imglist_augmix_dataset import ImglistAugMixDataset
from .imglist_extradata_dataset import ImglistExtraDataDataset, TwoSourceSampler
from .udg_dataset import UDGDataset


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)

        if split_config.dataset_class == 'ImglistExtraDataDataset':
            dataset = ImglistExtraDataDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor,
                extra_data_pth=split_config.extra_data_pth,
                extra_label_pth=split_config.extra_label_pth,
                extra_percent=split_config.extra_percent)

            batch_sampler = TwoSourceSampler(dataset.orig_ids,
                                             dataset.extra_ids,
                                             split_config.batch_size,
                                             split_config.orig_ratio)

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=dataset_config.num_workers,
            )
        elif split_config.dataset_class == 'ImglistAugMixDataset':
            dataset = ImglistAugMixDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
        else:
            CustomDataset = eval(split_config.dataset_class)
            dataset = CustomDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_tta_ood_dataloader(config: Config):
    ood_config = config.ood_dataset
    # specify custom dataset class
    CustomDataset = eval(ood_config.dataset_class)
    ood_period = config.pipeline.ood_period
    chunk_size = config.pipeline.chunk_size
    ood_ratio = config.pipeline.ood_ratio
    if ood_period < 0:
        ood_period = np.inf

    IOSampler = functools.partial(IDOODSampler,
                                  ood_ratio=ood_ratio,
                                  ood_period=max(1, ood_period))

    ind_dataset = get_dataloader(config)['test'].dataset
    ind_dataset = MixtureDataset(**{config.dataset.name: ind_dataset})
    ind_val_dataset = get_dataloader(config)['val'].dataset
    ind_val_dataset = MixtureDataset(**{config.dataset.name+'_val': ind_val_dataset})
    IODataset = functools.partial(IDOODDataset, ind=ind_dataset)
    pad_aux_sizes = {_: int(config.postprocessor.padding.get(_, 0) * chunk_size)
                     for _ in ('id', 'ood')}

    for _, s in pad_aux_sizes.items():
        if s < 0:  # -1: stratified
            pad_aux_sizes[_] = ood_config.batch_size

    dset_dict = {}
    dataloader_dict = {'aux': {'id': DataLoader(get_dataloader(config)['train'].dataset,
                                                shuffle=True,
                                                num_workers=ood_config.num_workers,
                                                drop_last=True,
                                                batch_size=pad_aux_sizes['id'])}}

    data_aux_preprocessor = TestStandardPreProcessor(config)

    if ood_config.padding and ood_config.padding.datasets:
        padding_config = ood_config['padding']
        padding_subssets = {}
        for dataset_name in padding_config.datasets:
            dataset_config = padding_config[dataset_name]
            dataset = CustomDataset(
                name=ood_config.name + '_padding_' + dataset_name,
                imglist_pth=dataset_config.imglist_pth,
                data_dir=dataset_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=get_preprocessor(config, split='padding.{}'.format(dataset_name)),
                data_aux_preprocessor=data_aux_preprocessor)
            padding_subssets[dataset_name] = dataset

        padding_set = MixtureDataset(**padding_subssets)
        if pad_aux_sizes['ood'] > 0.:
            dataloader_dict['aux']['ood'] = DataLoader(padding_set,
                                                       sampler=MixtureTimeSampler(
                                                           padding_set, period=1),
                                                       num_workers=ood_config.num_workers,
                                                       drop_last=True,
                                                       batch_size=pad_aux_sizes['ood'])

    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        if split == 'val':
            # validation set
            name = ood_config.name + '_' + split
            suboods = {}
            for dataset_name in split_config.datasets:
                if dataset_name == 'padding':
                    subset_seed = config.pipeline.seed
                    subset_indices = torch.randperm(len(padding_set),
                                                    generator=torch.Generator().manual_seed(subset_seed))

                    suboods[dataset_name] = Subset(padding_set, subset_indices[:len(ind_val_dataset)])

                else:
                    dataset_config = split_config[dataset_name]
                    suboods[dataset_name] = CustomDataset(name=name,
                                                          imglist_pth=dataset_config.imglist_pth,
                                                          data_dir=dataset_config.data_dir,
                                                          num_classes=ood_config.num_classes,
                                                          preprocessor=preprocessor,
                                                          data_aux_preprocessor=data_aux_preprocessor)
            dataset = IODataset(**suboods, ind=ind_val_dataset)
            dataloader = DataLoader(dataset,
                                    batch_size=chunk_size,
                                    num_workers=ood_config.num_workers,
                                    sampler=IOSampler(dataset))

            dataloader_dict[split] = dataloader
            continue

        if split == 'mixture':
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                suboods = {_: dset_dict[_] for _ in split_config[dataset_name].sets}
                dataset = IODataset(**suboods)
                dataloader = DataLoader(dataset,
                                        batch_size=chunk_size,
                                        num_workers=ood_config.num_workers,
                                        sampler=IOSampler(dataset))
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict
            continue

        if split in ('nearood', 'farood'):
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                ood_dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)

                dataset = IODataset(**{dataset_name: ood_dataset})
                dataloader = DataLoader(dataset,
                                        batch_size=chunk_size,
                                        num_workers=ood_config.num_workers,
                                        sampler=IOSampler(dataset))

                dset_dict[dataset_name] = ood_dataset
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict
            continue
        else:
            raise ValueError('{} split unknown'.format(split))

    return dataloader_dict


def get_ood_dataloader(config: Config):
    if config.pipeline.name == 'test_tta_ood':
        return get_tta_ood_dataloader(config)
    # specify custom dataset class for tta_ood
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader


def get_feature_opengan_dataloader(dataset_config: Config):
    feat_root = dataset_config.feat_root

    dataloader_dict = {}
    for d in ['id_train', 'id_val', 'ood_val']:
        # load in the cached feature
        loaded_data = load(os.path.join(feat_root, f'{d}.npz'),
                           allow_pickle=True)
        total_feat = torch.from_numpy(loaded_data['feat_list'])
        total_labels = loaded_data['label_list']
        del loaded_data
        # reshape the vector to fit in to the network
        total_feat.unsqueeze_(-1).unsqueeze_(-1)
        # let's see what we got here should be something like:
        # torch.Size([total_num, channel_size, 1, 1])
        print('Loaded feature size: {}'.format(total_feat.shape))

        if d == 'id_train':
            split_config = dataset_config['train']
        else:
            split_config = dataset_config['val']

        dataset = FeatDataset(feat=total_feat, labels=total_labels)
        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)
        dataloader_dict[d] = dataloader

    return dataloader_dict
