import os
import setGPU
import numpy as np
import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import vae.vae_particle as vap
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator as dage

# ********************************************************
#       runtime params
# ********************************************************

run_n = 103
beta = 0.01
loss = losses.make_threeD_kl_loss #losses.make_mse_kl_loss
reco_loss = losses.threeD_loss #losses.mse_loss
experiment = expe.Experiment(run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       prepare training set generator
# ********************************************************
sample_id = 'qcdSideAll'
data_generator = dage.DataGenerator(paths.sample_dir_path(sample_id))
mean_stdev = data_generator.get_mean_and_stdev(train_evts)

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=(100,3), z_sz=10, filter_ini_n=6, kernel_sz=3, loss=loss, reco_loss=reco_loss, batch_sz=128, beta=beta, regularizer='L1L2')
vae.build(mean_stdev)

# *******************************************************
#                       train and save
# *******************************************************

vae.fit(tf.data.Dataset.from_generator(data_generator, output_shapes=(None,100,3)), epochs=300, verbose=2)
vae.plot_training(experiment.fig_dir)
vae.save(path=experiment.model_dir)