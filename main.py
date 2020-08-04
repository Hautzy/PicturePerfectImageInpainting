import net
import config as c
import utils.data_visualization_utils as dv
import utils.preprocessing_utils as pp

c.clear_folders()
c.create_folders()
pp.scale_and_rotate_raw_images()
pp.create_train_samples()

#dv.test_set_info()

net.train()
#test.create_test_data()
#print(test.scoring(prediction_file='test/predictions.pk', target_file='test/example_targets.pkl'))
#gan.train()

