# ************************************************** #
# create datasets and dataloaders
# ************************************************** #
import torch
import numpy as np
import config as c
from pickle import load
import utils.normalization_utils as norm
from torch.utils.data import Dataset, Subset, DataLoader

from utils.preprocessing_utils import calculate_coordinates





# create train, test and validation data sets
def create_train_test(test_ratio=0.1, val_ratio=0.1):
    dataset = CropDataset(c.SAMPLES_FOLDER)
    n_samples = len(dataset)
    shuffled_indices = np.arange(n_samples)  # shuffled_indices = np.random.permutation(n_samples)
    test_set_indices = shuffled_indices[:int(n_samples * test_ratio)]
    val_set_indices = shuffled_indices[int(n_samples * test_ratio):int(n_samples * test_ratio + n_samples * val_ratio)]
    train_set_indices = shuffled_indices[int(n_samples * test_ratio + n_samples * val_ratio):]

    test_set = Subset(dataset, indices=test_set_indices)
    val_set = Subset(dataset, indices=val_set_indices)
    train_set = Subset(dataset, indices=train_set_indices)

    normalizer = norm.load_image_normalizer()
    if normalizer is None:
        normalizer = norm.GlobalImageNormalizer()
        normalizer.calculate_global_pixel_mean(train_set)

    return test_set, val_set, train_set, normalizer


# create train, test and validation data loader
def create_data_loader():
    test_set, val_set, train_set, normalizer = create_train_test()
    test_batch_size = 1
    val_batch_size = train_batch_size = 64

    test_loader = CropDataLoader(test_set, normalizer, batch_size=test_batch_size, num_workers=0)
    val_loader = CropDataLoader(val_set, normalizer, batch_size=val_batch_size, num_workers=16)
    train_loader = CropDataLoader(train_set, normalizer, batch_size=train_batch_size, num_workers=16)

    return test_loader, val_loader, train_loader, test_batch_size, val_batch_size, train_batch_size


class CropDataLoader(DataLoader):
    def __init__(self, data_set, normalizer, batch_size=1, num_workers=0):
        super(CropDataLoader, self).__init__(data_set, shuffle=False, batch_size=batch_size, num_workers=num_workers, collate_fn=self.stack_cropped_pictures)
        self.normalizer = normalizer

    # stack method for creating mini batches
    def stack_cropped_pictures(self, batch_as_list):
        batch_len = len(batch_as_list)
        X = torch.zeros(size=(batch_len, 1,
                              c.MAX_IMAGE_SIZE, c.MAX_IMAGE_SIZE))
        y = torch.zeros(size=(batch_len, 1, c.MAX_IMAGE_SIZE, c.MAX_IMAGE_SIZE))
        meta = list()
        targets = list()
        for ind in range(batch_len):
            crop_size = batch_as_list[ind]['crop_size']
            crop_center = batch_as_list[ind]['crop_center']
            image = batch_as_list[ind]['image']
            image = self.normalizer.transform(image)
            map = batch_as_list[ind]['map']
            target = batch_as_list[ind]['target']
            target = self.normalizer.transform(target)

            ih, iw = image.shape
            X[ind, 0, 0:ih, 0:iw] = image
            # X[ind, 1, :ih, :iw] = map

            st_x, en_x, st_y, en_y = calculate_coordinates(crop_size, crop_center)

            y[ind, 0, 0:ih, 0:iw] = image
            y[ind, 0, st_y:en_y, st_x:en_x] = target
            meta.append({
                'crop_size': crop_size,
                'crop_center': crop_center
            })
            targets.append(target)

        return X, y, meta, targets


# basic data loader for preprocessed samples
# loads real samples on demand!
class CropDataset(Dataset):
    def __init__(self, folder):
        self.sample_paths = c.get_file_paths(folder)
        self.sample_paths.sort()
        print(f'>>> loaded {len(self.sample_paths)} sample paths')

    def __getitem__(self, index):
        with open(self.sample_paths[index], 'rb') as f:
            return load(f)
        raise FileNotFoundError()

    def __len__(self):
        return len(self.sample_paths)
