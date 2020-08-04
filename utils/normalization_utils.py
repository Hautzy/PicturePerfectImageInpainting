import os
from pickle import load, dump

import config as c
import numpy as np
from torch.utils.data.dataset import Subset


def load_image_normalizer():
    if os.path.exists(c.PREPRO_IMAGE_NORMALIZER):
        with open(c.PREPRO_IMAGE_NORMALIZER, 'rb') as f:
            return load(f)
    return None


class GlobalImageNormalizer:

    def __init__(self):
        self.global_pixel_mean = 0.0

    def calculate_global_pixel_mean(self, subset: Subset):
        print('Try to find global mean pixel value')
        indices = subset.indices
        dataset = subset.dataset
        paths = [dataset.sample_paths[ind] for ind in indices]

        sample_num = len(indices)
        means = np.zeros(shape=(sample_num))
        for i, sample_path in enumerate(paths):
            with open(sample_path, 'rb') as f:
                img = load(f)['image']
                means[i] = img.double().mean()
        self.global_pixel_mean = means.mean()
        self.save_image_normalizer()
        print(f'>>> global mean pixel value is {self.global_pixel_mean}')
        return self.global_pixel_mean

    def transform(self, img):
        normalized_img = img - self.global_pixel_mean
        normalized_img = normalized_img / 255.0
        return normalized_img

    def transform_inv(self, img):
        inv_img = img * 255.0
        inv_img = inv_img + self.global_pixel_mean
        return inv_img

    def save_image_normalizer(self):
        with open(c.PREPRO_IMAGE_NORMALIZER, 'wb') as f:
            dump(self, f)
