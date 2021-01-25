import os
import setGPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import util.data_generator as dage
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_input as sdi


paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
data_train_generator = dage.DataGenerator(path=paths.sample_dir_path('qcdSide'), sample_part_n=int(3e5), sample_max_n=int(4e5)) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_types=tf.float32, output_shapes=(100,3)).batch(int(3e5)) # already shuffled

for dat in train_ds:
    print('new batch')