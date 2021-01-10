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
import training as tra


# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'run_n input_shape beta epochs train_total_n gen_part_n valid_total_n batch_n z_sz lambda_reg')
params = Parameters(run_n=107, input_shape=(100,3), beta=0.01, epochs=400, train_total_n=int(10e6), valid_total_n=int(1e6), gen_part_n=int(5e5), batch_n=512, z_sz=10, lambda_reg=0.0) # 'L1L2'
experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
data_train_generator = dage.DataGenerator(path=paths.sample_dir_path('qcdSide'), sample_part_n=params.gen_part_n, sample_max_n=params.train_total_n) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_types=tf.float32, output_shapes=params.input_shape).batch(params.batch_n, drop_remainder=True) # already shuffled

# validation (full tensor, 1M events -> 2M samples)
data_valid = dage.constituents_to_input_samples(dare.DataReader(path=paths.sample_dir_path('qcdSideExt')).read_constituents_from_dir(read_n=params.valid_total_n))
valid_ds = tf.data.Dataset.from_tensor_slices(data_valid).batch(params.batch_n, drop_remainder=True)

# stats for normalization layer
mean_stdev = data_train_generator.get_mean_and_stdev()

# *******************************************************
#                       training options
# *******************************************************

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = losses.threeD_loss

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=params.input_shape, z_sz=params.z_sz, filter_ini_n=6, kernel_sz=3)
vae.build(mean_stdev)

# *******************************************************
#                       train and save
# *******************************************************

trainer = tra.Trainer(optimizer=optimizer, beta=params.beta, patience=4, min_delta=0.01, max_lr_decay=5, lambda_reg=params.lambda_reg)
losses_reco, losses_valid = trainer.train(vae=vae, loss_fn=loss_fn, train_ds=train_ds, valid_ds=valid_ds, epochs=params.epochs, model_dir=experiment.model_dir)
tra.plot_training_results(losses_reco, losses_valid, experiment.fig_dir)

vae.save(path=experiment.model_dir)
