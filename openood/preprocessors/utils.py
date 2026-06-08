from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cider_preprocessor import CiderPreprocessor
from .csi_preprocessor import CSIPreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .augmix_preprocessor import AugMixPreprocessor
from .pixmix_preprocessor import PixMixPreprocessor
from .randaugment_preprocessor import RandAugmentPreprocessor
from .cutout_preprocessor import CutoutPreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .palm_preprocessor import PALMPreprocessor
from .patch_preprocessor import PatchPreprocessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'augmix': AugMixPreprocessor,
        'pixmix': PixMixPreprocessor,
        'randaugment': RandAugmentPreprocessor,
        'cutout': CutoutPreprocessor,
        'csi': CSIPreprocessor,
        'cider': CiderPreprocessor,
        'palm': PALMPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
    }

    if split == 'train':
        return train_preprocessors.get(config.preprocessor.name, 'base')(config)
    elif split.startswith('padding.'):
        return PatchPreprocessor(config, split=split)
    else:
        return test_preprocessors.get(config.preprocessor.name, 'base')(config)
