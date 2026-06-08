import random
import torch
import torchvision.transforms as tvs_trans
from torchvision.transforms import Lambda
import torchvision.transforms.v2 as tvs_trans_v2

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict
from .test_preprocessor import TestStandardPreProcessor

from collections import deque


class RandomPatch:

    def __call__(self, img):

        C, H, W = img.shape

        for _ in range(random.randint(1, 4)):
            size = random.randint(3, 12)
            mode = random.choice(('black', 'noise', 'color', 'shuffle'))
            patch = self._create_patch(img, size, mode)

            x = random.randint(0, H - size)
            y = random.randint(0, W - size)

            img[:, x:x+size, y:y+size] = patch

        return img

    def _create_patch(self, img, size, mode):
        C, H, W = img.shape
        if mode == 'black':
            patch = torch.zeros((C, size, size), device=img.device)
        elif mode == 'noise':
            patch = torch.randn((C, size, size), device=img.device)
        elif mode == 'color':
            patch = torch.rand((C, 1, 1), device=img.device).expand(C, size, size)*2-1
        else:  # shuffle patch
            y2 = random.randint(0, H-size)
            x2 = random.randint(0, W-size)
            patch = img[:, y2:y2+size, x2:x2+size].clone()

        return patch

    def __repr__(self):
        return 'Patch'


class Cutout(RandomPatch):

    def __call__(self, img):

        C, H, W = img.shape

        size = random.randint(4, 10)
        mode = random.choice(('black', 'noise'))
        patch = self._create_patch(img, size, mode)

        x = random.randint(0, H - size)
        y = random.randint(0, W - size)

        img[:, x:x+size, y:y+size] = patch

        return img

    def __repr__(self):
        return 'Cutout'


class Mixup:

    def __init__(self, max_alpha=0.3):

        # self._buffer = deque(maxlen=buffer_size)
        self._lastone = None
        self._max_alpha = max_alpha

    def __call__(self, img):

        img2 = self._lastone
        self._lastone = img

        if img2 is None:
            return img

        alpha = random.uniform(0, self._max_alpha)

        return (1 - alpha) * img + alpha * img2

    def __repr__(self):

        return 'Mixup'


class GaussianNoise:

    def __init__(self, mean=0, sigma=0.1, clip=True):

        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def __call__(self, img):

        noise = torch.randn_like(img)

        img_ = img + self.mean + self.sigma * noise
        if not self.clip:
            return img_

        return img_.clip(0, 1)

    def __repr__(self):

        return 'GaussianNoise(mean={}, std={}{})'.format(self.mean, self.sigma, ', <>' if self.clip else '')


class PatchPreprocessor(TestStandardPreProcessor):
    """For patched dataset for outlier exposure"""

    def __init__(self, config: Config, split=None):
        super().__init__(config)
        if not split:
            return

        dataset_name = split.split('.')[-1]
        preprocessor_config = config.ood_dataset.padding.get(dataset_name, {}).get('preprocessor', {})
        preprocessor_name = preprocessor_config.get('name', 'base')
        if preprocessor_name == 'base':
            return

        self.preprocessor_args = preprocessor_config.get('preprocessor_args', {})

        for p in self.preprocessor_args:
            if p == 'cutout':
                t = Cutout()
            elif p == 'patch':
                t = RandomPatch()
            elif p == 'mixup':
                t = Mixup()
            elif p == 'noise':
                t = GaussianNoise(0, 0.05, clip=True)
            elif p == 'blur':
                t = tvs_trans.RandomApply([tvs_trans.GaussianBlur((3, 3),
                                                                  (0.1, 1.0))],
                                          p=0.5)
            elif p == 'geometric':
                t = tvs_trans.RandomAffine(15, translate=(0.1, 0.1))
            else:
                raise NotImplementedError('{} transform'.format(p))

            # insert t before normalization
            self.transform.transforms.insert(-1, t)


if __name__ == '__main__':

    #     from torchvision.transforms.v2 import GaussianNoise

    from pathlib import Path
    config_dir = Path('configs') / 'datasets' / 'cifar10'

    config = Config(str(config_dir / 'cifar10.yml'), str(config_dir / 'cifar10_tta_ood.yml'))

    p = PatchPreprocessor(config, 'padding.lsunr')
    p = PatchPreprocessor(config, 'padding.patched_id')

    print(p)
