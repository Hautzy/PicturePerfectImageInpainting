import net
import test
from pickle import load, HIGHEST_PROTOCOL, dump

import torch
from torch import nn

import config as c
import numpy as np
from gan import gan
import crop_dataset as data
import preprocessing as pp

#c.create_folders()
#pp.scale_rotate_images()
#pp.create_train_samples()

net.train()
#test.create_test_data()
#print(test.scoring(prediction_file='test/predictions.pk', target_file='test/example_targets.pkl'))

#gan.train()