# ************************************************** #
# methods testing models on the given testing data
# ************************************************** #
import os
from pickle import dump, load

import torch
import dill as pkl
import numpy as np
import config as c
from utils import preprocessing_utils as pp
from utils.preprocessing_utils import calculate_coordinates


# calculate MSE from predicted cropped out sub-image and the correct test sub-image
def mse(target_array, prediction_array, ind):
    if prediction_array.shape != target_array.shape:
        raise IndexError(
            f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}. Prediction {ind}")
    prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
    return np.mean((prediction_array - target_array) ** 2)


# create mean MSE for all test samples based on the prediction file and target file
def scoring(prediction_file: str, target_file: str):
    with open(prediction_file, 'rb') as pfh:
        predictions = pkl.load(pfh)
    with open(target_file, 'rb') as tfh:
        targets = pkl.load(tfh)

    mses = np.zeros(shape=len(predictions))
    ind = 0
    for target, prediction in zip(targets, predictions):
        mses[ind] = mse(target, prediction, ind)
        ind += 1

    return np.mean(mses)


# test best model over all given test data from test data pickle file and save results for later scoring
def create_test_data():
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
        image_array, crop_array, target_array = pp.create_crop_from_single_image(image, crop_size, crop_center)
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