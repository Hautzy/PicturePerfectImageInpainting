import os

from torch.utils.data import DataLoader

import config as c
import preprocessing as pp
from pickle import dump, HIGHEST_PROTOCOL, load

import torch

from crop_dataset import CropDataset, stack_cropped_pictures, calculate_coordinates
from net import evaluate_model


def run_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(c.TEST_FOLDER + os.sep + 'example_testset.pkl', 'rb') as f:
        test_set = load(f)

    crop_sizes = test_set['crop_sizes']
    crop_centers = test_set['crop_centers']
    images = test_set['images']
    targets = list()
    model = torch.load(c.BEST_MODEL_FILE)

    for ind in range(len(images)):
        crop_size = crop_sizes[ind]
        crop_center = crop_centers[ind]
        image = images[ind]
        image_array, crop_array, target_array = pp.crop_from_image(image, crop_size, crop_center)
        X = torch.tensor(image_array)
        new_X = torch.zeros(size=(1, 1, c.MAX_IMAGE_SIZE, c.MAX_IMAGE_SIZE), device=device)
        ih, iw = image_array.shape
        new_X[0, 0, :ih, :iw] = X
        output = model(new_X)
        st_x, en_x, st_y, en_y = calculate_coordinates(crop_size, crop_center)
        gpu_target = output[0, 0, st_y:en_y, st_x:en_x]
        cpu_target = gpu_target.cpu().detach()
        py_target = cpu_target.numpy().astype('uint8')
        targets.append(py_target)
        print(f'>>> tested sample {ind + 1}')
    with open(c.TEST_TARGETS_FILE, 'wb') as f:
        dump(targets, f)