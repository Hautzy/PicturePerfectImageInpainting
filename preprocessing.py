import os
import torch
import random
import numpy as np
import config as c
from PIL import Image
from pickle import dump, HIGHEST_PROTOCOL, load
from crop_dataset import calculate_coordinates


def scale_rotate_images():
    index = 0
    raw_image_paths = c.get_file_paths(c.RAW_DATA_FOLDER)[:100]
    print('### SCALE IMAGES ###')
    for image_path in raw_image_paths:
        image = Image.open(image_path)
        image.thumbnail((c.MAX_IMAGE_SIZE, c.MAX_IMAGE_SIZE), Image.ANTIALIAS)
        deg = 0
        for ind in range(c.NUMBER_OF_ROTATIONS):
            if deg != 0:
                rotated_image = image.rotate(deg, fillcolor=0)
            else:
                rotated_image = image
            file = f'{c.PREPRO_IMAGES_FOLDER}/i{index}_r{deg}.jpg'
            rotated_image.save(file)
            print(f'>>> file "{file}" created')
            deg += c.FIXED_ROTATION
            index += 1


def crop_from_image(image_array, crop_size, crop_center):
    if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 2:
        raise ValueError('No numpy 2d array')
    elif len(crop_size) != 2 or len(crop_center) != 2:
        raise ValueError('tuples not size 2')
    elif crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
        raise ValueError('crop size not odd')
    ih, iw = image_array.shape
    st_x, en_x, st_y, en_y = calculate_coordinates(crop_size, crop_center)
    if st_x < 20 or iw - en_x < 20 or st_y < 20 or ih - en_y < 20:
        raise ValueError('rectangle is < 20 pixels from borders')

    target_array = np.copy(image_array[st_y:en_y, st_x:en_x])
    image_array[st_y:en_y, st_x:en_x] = 0
    crop_array = np.zeros(shape=image_array.shape, dtype=image_array.dtype)
    crop_array[st_y:en_y, st_x:en_x] = 1

    return image_array, crop_array, target_array


def get_random_crop_size_and_center(image_height, image_width):
    valid_crop = False
    trys = 0
    while not valid_crop and trys < 10:
        crop_width = random.randrange(c.MIN_CROP_SIZE, c.MAX_CROP_SIZE + 1, 2)
        crop_height = random.randrange(c.MIN_CROP_SIZE, c.MAX_CROP_SIZE + 1, 2)

        start_x = c.MIN_PADDING + int(crop_width / 2) + 1
        start_y = c.MIN_PADDING + int(crop_height / 2) + 1
        end_x = image_width - c.MIN_PADDING - int(crop_width / 2) - 1
        end_y = image_height - c.MIN_PADDING - int(crop_height / 2) - 1
        valid_crop = end_y-start_y > 0 and end_x-start_x > 0
        trys += 1

    if not valid_crop:
        return None, None
    center_x = random.randrange(start_x, end_x)
    center_y = random.randrange(start_y, end_y)
    return (crop_height, crop_width), (center_y, center_x)


def create_train_samples():
    index = 0
    image_paths = c.get_file_paths(c.PREPRO_IMAGES_FOLDER)
    print('### CREATE SAMPLES ###')
    for image_path in image_paths:
        image = Image.open(image_path)
        for i in range(c.CROPS_PER_PICTURE):
            image_array = np.array(image, dtype='uint8')
            crop_size, crop_center = get_random_crop_size_and_center(image_array.shape[0], image_array.shape[1])
            if crop_size is None or crop_center is None:
                continue
            image_array, crop_array, target_array = crop_from_image(image_array, crop_size, crop_center)

            sample = {
                'crop_size': crop_size,
                'crop_center': crop_center,

                'image': torch.tensor(image_array),
                'map': torch.tensor(crop_array),
                'target': torch.tensor(target_array)
            }
            file = f'{c.SAMPLES_FOLDER}/{index}.pk'
            with open(file, 'wb') as f:
                dump(sample, f, protocol=HIGHEST_PROTOCOL)
            print(f'>>> file "{file}" created')
            index += 1