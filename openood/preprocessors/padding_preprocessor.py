import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict
from .test_preprocessor import TestStandardPreProcessor


class PaddingPreprocessor(TestStandardPreProcessor):
    """For patched dataset for outlier exposure"""

    def __init__(self, config: Config):
        super().__init__(config)

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        norm = self.transform.transforms[-1]
        m, s = norm.mean, norm.std
        # return '{} / {}'.format(m, s)
        return ''.join(map(str.strip, self.transform.__repr__().split('\n')))
