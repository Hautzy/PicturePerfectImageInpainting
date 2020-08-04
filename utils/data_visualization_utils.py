import os
import shutil
import numpy as np
import pickle as pkl
from PIL import Image

import config as c


def sample_to_human_readable(sample_id):
    with open(c.SAMPLES_FOLDER + os.sep + sample_id + '.pk', 'rb') as pfh:
        sample = pkl.load(pfh)
    sample_folder = c.OUT_FOLDER + os.sep + sample_id
    os.mkdir(sample_folder)

    crop_size = sample['crop_size']
    crop_center = sample['crop_center']

    with open(sample_folder + os.sep + 'meta.txt', 'w') as fh:
        print(f"crop_size: {crop_size}", file=fh)
        print(f"crop_center: {crop_center}", file=fh)

    image = sample['image']
    map = sample['map']
    target = sample['target']

    image = image.numpy()
    map = map.numpy()
    target = target.numpy()

    image = Image.fromarray(image)
    image.save(sample_folder + os.sep + 'image.png')

    map = Image.fromarray(map)
    map.save(sample_folder + os.sep + 'map.png')

    target = Image.fromarray(target)
    target.save(sample_folder + os.sep + 'target.png')


def test_set_to_human_readable():
    with open(c.TEST_FOLDER + os.sep + 'example_testset.pkl', 'rb') as f:
        test_set = pkl.load(f)
    with open(c.TEST_FOLDER + os.sep + 'example_targets.pkl', 'rb') as f:
        targets = pkl.load(f)
    test_folder = c.OUT_FOLDER + os.sep + 'test_set'
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder, ignore_errors=True)
    os.mkdir(test_folder)

    crop_sizes = test_set['crop_sizes']
    crop_centers = test_set['crop_centers']
    images = test_set['images']

    for i in range(len(targets)):
        current_sample_folder = test_folder + os.sep + str(i)
        os.mkdir(current_sample_folder)

        with open(current_sample_folder + os.sep + 'meta.txt', 'w') as fh:
            print(f"crop_size: {crop_sizes[i]}", file=fh)
            print(f"crop_center: {crop_centers[i]}", file=fh)

        img = Image.fromarray(images[i])
        img.save(current_sample_folder + os.sep + 'image.png')

        target = Image.fromarray(targets[i])
        target.save(current_sample_folder + os.sep + 'target.png')

