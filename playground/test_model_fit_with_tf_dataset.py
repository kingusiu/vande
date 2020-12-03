import os
import setGPU
import tensorflow as tf
import numpy as np

import pofah.path_constants.sample_dict_file_parts_input as sdi
import sarewt.data_reader as dare
import util.data_generator as dage
import utilities as ut

# number setup
samples_max_n = int(1e6) # was 1m
batch_sz = 256
steps_per_epoch = samples_max_n // batch_sz 
events_valid_n = int(1e3) # was 1e3
steps_valid_per_epoch = events_valid_n*2 // batch_sz #(n events a 2 jets = 2n inputs)
gen_part_train_n = int(1e3) # was 1e4
gen_part_valid_n = int(1e3) # was 1e2

# data generator
path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir']['qcdSide'])
gen = dage.DataGenerator(path=path, sample_part_n=gen_part_train_n, sample_max_n=samples_max_n) # generate samples_max_n jet samples
# tf dataset from generator
tfds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32), output_shapes=((100,3),(100,3))).batch(batch_sz, drop_remainder=True)

# validation data
path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir']['qcdSideExt'])
#gen_valid = dage.DataGenerator(path=path, samples_in_parts_n=gen_part_train_n, samples_max_n=events_valid_n)
data_valid = dage.constituents_to_input_samples(dare.DataReader(path=path).read_constituents_from_dir(read_n=events_valid_n))
tfds_valid = tf.data.Dataset.from_tensor_slices((data_valid, data_valid)).batch(batch_sz, drop_remainder=True)

# DNN
model = ut.get_simple_autoencoder()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN()]
model.fit(tfds, epochs=100, verbose=2, validation_data=tfds_valid, callbacks=callbacks)
# model.fit(tfds, epochs=100, verbose=2, validation_data=(x_valid, x_valid), validation_steps=10)
