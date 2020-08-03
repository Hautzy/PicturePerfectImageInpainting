# ************************************************** #
# basic utils and constants for configuration
# ************************************************** #
import os

RAW_DATA_FOLDER = 'raw_data'
PREPRO_FOLDER = 'pre_pro'
MODEL_FOLDER = 'model'
TEST_FOLDER = 'test'
OUT_FOLDER = 'out'

PREPRO_IMAGES_FOLDER = PREPRO_FOLDER + os.sep + 'images'
SAMPLES_FOLDER = PREPRO_FOLDER + os.sep + 'samples'
SCALER_FILE = PREPRO_FOLDER + os.sep + 'scaler.pk'
BEST_MODEL_FILE = MODEL_FOLDER + os.sep + 'best_model.pk'
BEST_RESULTS_FILE = MODEL_FOLDER + os.sep + 'best_results.txt'
TEST_TARGETS_FILE = TEST_FOLDER + os.sep + 'predictions.pk'

MIN_IMAGE_SIZE = 70
MAX_IMAGE_SIZE = 100

MIN_CROP_SIZE = 5
MAX_CROP_SIZE = 21

MIN_PADDING = 20

CROPS_PER_PICTURE = 10
NUMBER_OF_ROTATIONS = 8
FIXED_ROTATION = 45


def create_folders():
    if not os.path.exists(PREPRO_FOLDER):
        os.mkdir(PREPRO_FOLDER)
    if not os.path.exists(PREPRO_IMAGES_FOLDER):
        os.mkdir(PREPRO_IMAGES_FOLDER)
    if not os.path.exists(SAMPLES_FOLDER):
        os.mkdir(SAMPLES_FOLDER)
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    if not os.path.exists(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)
    print('>>> Created folder structure')


def get_file_paths(root_folder):
    paths = list()
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            paths.append(os.path.join(subdir, file))
    return paths
