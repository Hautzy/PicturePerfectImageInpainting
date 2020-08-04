import os
import config as c
from PIL import Image

img_path = 'raw_data/dataset_part_0/0000/0001.jpg'
img = Image.open(img_path)
size = img.size

img.thumbnail((100, 100), Image.ANTIALIAS)
img.save(c.OUT_FOLDER + os.sep + 'test.jpg')