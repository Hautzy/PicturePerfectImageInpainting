import net
import test
from pickle import load, HIGHEST_PROTOCOL, dump

import torch
from torch import nn

import config as c
import numpy as np
import crop_dataset as data
import preprocessing as pp
from scoring import scoring

#c.create_folders()
#pp.scale_rotate_images()
#pp.create_samples()

net.train()
test.run_tests()
mse_loss = scoring(prediction_file='test/predictions.pk', target_file='test/example_targets.pkl')#print(mse_loss)
print(mse_loss)