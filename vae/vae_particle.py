import os
#import setGPU
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt

import vae.losses as losses
import vae.vae_base as vbase


# custom 1d transposed convolution that expands to 2d output for vae decoder
class Conv1DTranspose(tf.keras.layers.Layer):

	def __init__(self, filters, kernel_sz, activation, regularizer=None, **kwargs):
		super(Conv1DTranspose,self).__init__(**kwargs)
		self.kernel_sz = kernel_sz
		self.filters = filters
		self.activation = activation
		self.regularizer = regularizer
		self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
		self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(self.kernel_sz,1), activation=self.activation, kernel_regularizer=self.regularizer)
		self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

	def call(self, inputs):
		# expand input and kernel to 2D
		x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
		# call Conv2DTranspose
		x = self.ConvTranspose(x)
		# squeeze back to 1D and return
		x = self.SqueezeChannel(x)
		return x

	def get_config(self):
		config = super(Conv1DTranspose, self).get_config()
		config.update({'kernel_sz': self.kernel_sz, 'filters': self.filters, 'activation': self.activation, 'regularizer': self.regularizer})
		return config


class VAEparticle(vbase.VAE):

	def __init__(self, input_shape=(100,3), z_sz=10, filter_ini_n=6, kernel_sz=3, regularizer=None):
		super().__init__(input_shape=input_shape, z_sz=z_sz, filter_ini_n=filter_ini_n, kernel_sz=kernel_sz, regularizer=regularizer)

	def build_encoder(self, mean, stdev):
		inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
		# normalize
		normalized = tf.keras.layers.Lambda(lambda xx: (xx-mean)/stdev)(inputs)
		# add channel dim
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(normalized) # [B x 100 x 3] => [B x 100 x 3 x 1]
		# 2D Conv
		x = tf.keras.layers.Conv2D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.regularizer)(x)
		# Squeeze
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x)  # remove width axis for 1D Conv [ B x 98 x 1 x filter_n ] -> [ B x 98 x filter_n ]
		# 1D Conv * 2
		self.filter_n += 4
		x = tf.keras.layers.Conv1D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.regularizer)(x) # [ B x 96 x 10 ]
		self.filter_n += 4
		x = tf.keras.layers.Conv1D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.regularizer)(x) # [ B x 94 x 14 ]
		# Pool
		x = tf.keras.layers.AveragePooling1D()(x) # [ B x 47 x 14 ]
		# shape info needed to build decoder model
		self.shape_convolved = x.get_shape().as_list()
		# Flatten
		x = tf.keras.layers.Flatten()(x) #[B x 658]
		# Dense * 3
		x = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce convolution output
		x = tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce again
		# x = Dense(8, activation='relu')(x)

		# *****************************
		#         latent space
		# generate latent vector Q(z|X)

		self.z_mean = tf.keras.layers.Dense(self.params.z_sz, name='z_mean', kernel_regularizer=self.regularizer)(x)
		self.z_log_var = tf.keras.layers.Dense(self.params.z_sz, name='z_log_var', kernel_regularizer=self.regularizer)(x)

		# use reparameterization trick to push the sampling out as input
		self.z = vbase.Sampling()((self.z_mean, self.z_log_var))

		# instantiate encoder model
		encoder = tf.keras.Model(inputs, [self.z, self.z_mean, self.z_log_var], name='encoder')
		encoder.summary()
		# plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
		return encoder

	def build_decoder(self, mean, stdev):
		latent_inputs = tf.keras.layers.Input(shape=(self.params.z_sz,), name='z_sampling')
		# Dense * 3
		x = tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
		x = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=self.regularizer)(x)  # double size
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]), activation='relu', kernel_regularizer=self.regularizer)(x)
		# Reshape
		x = tf.keras.layers.Reshape(tuple(self.shape_convolved[1:]))(x)
		# Upsample
		x = tf.keras.layers.UpSampling1D()(x) # [ B x 94 x 16 ]
		# 1D Conv Transpose * 2
		self.filter_n -= 4
		x = Conv1DTranspose(filters=self.filter_n, kernel_sz=self.params.kernel_sz, activation='relu', regularizer=self.regularizer)(x) # [ B x 94 x 16 ] -> [ B x 96 x 8 ]
		self.filter_n -= 4
		x = Conv1DTranspose(filters=self.filter_n, kernel_sz=self.params.kernel_sz, activation='relu', regularizer=self.regularizer)(x) # [ B x 96 x 8 ] -> [ B x 98 x 4 ]
		# Expand
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2))(x) #  [ B x 98 x 1 x 4 ]
		# 2D Conv Transpose
		x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.params.kernel_sz, activation=tf.keras.activations.elu, kernel_regularizer=self.regularizer, name='conv_2d_transpose')(x)
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x) # [B x 100 x 3 x 1] -> [B x 100 x 3]
		outputs_decoder = tf.keras.layers.Lambda(lambda xx: (xx*stdev)+mean, name='un_normalized_decoder_out')(x)

		# instantiate decoder model
		decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
		decoder.summary()
		# plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
		return decoder

	def fit(self, x_train, epochs=100, verbose=2, validation_data=None, validation_steps=100):
		callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN()] #TensorBoard(log_dir=self.log_dir, histogram_freq=1)
		if isinstance(x_train, np.ndarray): # if input is numpy array
			self.history = self.model.fit(x_train, x_train, epochs=epochs, batch_size=self.params.batch_sz, verbose=verbose, validation_split=0.25, callbacks=callbacks)
		else: # else if input is a generator
			print('calling generator fit()')
			self.history = self.model.fit(x_train, epochs=epochs, verbose=verbose, validation_data=validation_data, callbacks=callbacks)

	@classmethod
	def load(cls, path):
		custom_objects = {'Sampling': vbase.Sampling, 'Conv1DTranspose': Conv1DTranspose}
		return super().load(path=path, custom_objects=custom_objects)
