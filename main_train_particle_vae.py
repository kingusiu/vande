import os
import setGPU
import numpy as np
from collections import namedtuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import vae.vae_particle as vap
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator as dage
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as tra


# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'run_n input_shape kernel_sz kernel_ini_n beta epochs train_total_n gen_part_n valid_total_n batch_n z_sz activation initializer learning_rate max_lr_decay lambda_reg')
params = Parameters(run_n=113, 
                    input_shape=(100,3),
                    kernel_sz=(1,3), 
                    kernel_ini_n=12,
                    beta=0.01, 
                    epochs=400, 
                    train_total_n=int(10e6), 
                    valid_total_n=int(1e6), 
                    gen_part_n=int(5e5), 
                    batch_n=256, 
                    z_sz=12,
                    activation='elu',
                    initializer='he_uniform',
                    learning_rate=0.001,
                    max_lr_decay=8, 
                    lambda_reg=0.0) # 'L1L2'

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
print('>>> Preparing training dataset generator')
data_train_generator = dage.DataGenerator(path=paths.sample_dir_path('qcdSide'), sample_part_n=params.gen_part_n, sample_max_n=params.train_total_n, **cuts.global_cuts) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_types=tf.float32, output_shapes=params.input_shape).batch(params.batch_n, drop_remainder=True) # already shuffled

# validation (full tensor, 1M events -> 2M samples)
print('>>> Preparing validation dataset')
const_valid, _, features_valid, _ = dare.DataReader(path=paths.sample_dir_path('qcdSideExt')).read_events_from_dir(read_n=params.valid_total_n, **cuts.global_cuts)
data_valid = dage.events_to_input_samples(const_valid, features_valid)
valid_ds = tf.data.Dataset.from_tensor_slices(data_valid).batch(params.batch_n, drop_remainder=True)

# stats for normalization layer
mean_stdev = data_train_generator.get_mean_and_stdev()

# *******************************************************
#                       training options
# *******************************************************

optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
loss_fn = losses.threeD_loss

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=params.input_shape, z_sz=params.z_sz, kernel_ini_n=params.kernel_ini_n, kernel_sz=params.kernel_sz, activation=params.activation, initializer=params.initializer)
vae.build(mean_stdev)

# *******************************************************
#                       train and save
# *******************************************************
print('>>> Launching Training')
trainer = tra.Trainer(optimizer=optimizer, beta=params.beta, patience=3, min_delta=0.03, max_lr_decay=params.max_lr_decay, lambda_reg=params.lambda_reg)
losses_reco, losses_valid = trainer.train(vae=vae, loss_fn=loss_fn, train_ds=train_ds, valid_ds=valid_ds, epochs=params.epochs, model_dir=experiment.model_dir)
tra.plot_training_results(losses_reco, losses_valid, experiment.fig_dir)

vae.save(path=experiment.model_dir)
