import torch
import numpy as np
import config as c
from pickle import load
from torch.utils.data import Dataset, Subset, DataLoader


def stack_cropped_pictures(batch_as_list):
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
        map = batch_as_list[ind]['map']
        target = batch_as_list[ind]['target']

        ih, iw = image.shape
        X[ind, 0, :ih, :iw] = image
        #X[ind, 1, :ih, :iw] = map

        st_x, en_x, st_y, en_y = calculate_coordinates(crop_size, crop_center)

        y[ind, 0, :ih, :iw] = image
        y[ind, 0, st_y:en_y, st_x:en_x] = target
        meta.append({
            'crop_size': crop_size,
            'crop_center': crop_center
        })
        targets.append(target)

    return X, y, meta, targets


def calculate_coordinates(crop_size, crop_center):
    st_x = int(crop_center[1] - (crop_size[1] - 1) / 2)
    en_x = int(crop_center[1] + (crop_size[1] - 1) / 2) + 1
    st_y = int(crop_center[0] - (crop_size[0] - 1) / 2)
    en_y = int(crop_center[0] + (crop_size[0] - 1) / 2) + 1
    return st_x, en_x, st_y, en_y


def create_train_test(test_ratio=0.1, val_ratio=0.1):
    dataset = CropDataset(c.SAMPLES_FOLDER)
    n_samples = len(dataset)
    shuffled_indices = np.random.permutation(n_samples)
    test_set_indices = shuffled_indices[:int(n_samples * test_ratio)]
    val_set_indices = shuffled_indices[int(n_samples * test_ratio):int(n_samples * test_ratio + n_samples * val_ratio)]
    train_set_indices = shuffled_indices[int(n_samples * test_ratio + n_samples * val_ratio):]

    test_set = Subset(dataset, indices=test_set_indices)
    val_set = Subset(dataset, indices=val_set_indices)
    train_set = Subset(dataset, indices=train_set_indices)
    return test_set, val_set, train_set


def create_data_loader():
    test_set, val_set, train_set = create_train_test()
    test_batch_size = 1
    val_batch_size = train_batch_size = 64

    test_loader = DataLoader(test_set, shuffle=False, batch_size=test_batch_size, num_workers=0, collate_fn=stack_cropped_pictures)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=val_batch_size, num_workers=0, collate_fn=stack_cropped_pictures)
    train_loader = DataLoader(train_set, shuffle=False, batch_size=train_batch_size, num_workers=0, collate_fn=stack_cropped_pictures)

    return test_loader, val_loader, train_loader, test_batch_size, val_batch_size, train_batch_size


class CropDataset(Dataset):
    def __init__(self, folder):
        self.sample_paths = c.get_file_paths(folder)
        print(f'>>> loaded {len(self.sample_paths)} sample paths')

    def __getitem__(self, index):
        with open(self.sample_paths[index], 'rb') as f:
            return load(f)
        raise FileNotFoundError()

    def __len__(self):
        return len(self.sample_paths)
