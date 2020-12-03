import os
import setGPU
import numpy as np
from collections import namedtuple
import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import vae.vae_particle as vap
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator as dage
import sarewt.data_reader as dare


# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'run_n beta train_total_n gen_part_n valid_total_n batch_n regularizer')
params = Parameters(run_n=104, beta=0.01, train_total_n=int(10e6), valid_total_n=int(1e6), gen_part_n=int(1e5), batch_n=256, regularizer=None) # 'L1L2'
loss = losses.make_threeD_kl_loss #losses.make_mse_kl_loss
reco_loss = losses.threeD_loss #losses.mse_loss
experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
data_train_generator = dage.DataGenerator(path=paths.sample_dir_path('qcdSide'), sample_part_n=params.gen_part_n, sample_max_n=params.train_total_n) # generate 10 M jet samples
data_train = tf.data.Dataset.from_generator(data_train_generator, output_types=(tf.float32, tf.float32), output_shapes=((100,3),(100,3))).batch(params.batch_n, drop_remainder=True) # already shuffled

# validation (full tensor, 2M samples)
data_valid = dage.constituents_to_input_samples(dare.DataReader(path=paths.sample_dir_path('qcdSideExt')).read_constituents_from_dir(read_n=params.valid_total_n))
ds_valid = tf.data.Dataset.from_tensor_slices((data_valid, data_valid)).batch(params.batch_n, drop_remainder=True)

# stats for normalization layer
mean_stdev = data_train_generator.get_mean_and_stdev()

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=(100,3), z_sz=10, filter_ini_n=6, kernel_sz=3, loss=loss, reco_loss=reco_loss, batch_sz=params.batch_n, beta=params.beta, regularizer=params.regularizer)
vae.build(mean_stdev)

# *******************************************************
#                       train and save
# *******************************************************

vae.fit(data_train, epochs=300, validation_data=ds_valid, verbose=2)
vae.plot_training(experiment.fig_dir)
vae.save(path=experiment.model_dir)