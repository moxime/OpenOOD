import torch
import random
from torchvision.datasets import LSUN
from pathlib import Path as P
from torchvision import transforms as t
from torch.utils.data import DataLoader


resize_t = t.Compose([t.Resize((32, 32))])
root = P('~/dev/dnn/torch-data/lsun').expanduser()
lsunr = LSUN(root=root, classes='test', transform=resize_t)


dict_of_sets = {}


dict_of_sets['lsunr'] = {'split': 'test',
                         'dset': lsunr,
                         'data_dir': P('./data/images_classic/'),
                         'imglist_pth': P('./data/benchmark_imglist/cifar10/test_lsunr.txt')}


if __name__ == '__main__':

    dset = 'lsunr'

    d = dict_of_sets[dset]
    data_dir = d['data_dir']
    split = d['split']

    list_of_path = []

    for i, (im, y) in enumerate(lsunr):
        impath = P(dset) / split / '{:04d}.jpg'.format(i)
        # im.save(data_dir / impath, 'JPEG')
        list_of_path.append(impath)

    random.shuffle(list_of_path)
    with open(d['imglist_pth'], 'w') as f:
        for p in list_of_path:
            f.write(str(p) + ' -1' + '\n')
