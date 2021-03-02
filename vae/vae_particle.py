import os
#import setGPU
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt

import vae.losses as losses
import vae.vae_base as vbase
import vae.layers as layers



class VAEparticle(vbase.VAE):

	def __init__(self, input_shape=(100,3), z_sz=10, kernel_ini_n=6, kernel_sz=3, kernel_1D_sz=3, beta=0.01, activation='relu', initializer='glorot_uniform'):
		super().__init__(input_shape=input_shape, z_sz=z_sz, kernel_ini_n=kernel_ini_n, kernel_sz=kernel_sz, kernel_1D_sz=3, beta=beta, activation=activation, initializer=initializer)

	def build_encoder(self, mean, stdev):
		inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
		# normalize
		normalized = layers.StdNormalization(mean_x=mean, std_x=stdev)(inputs)
		# add channel dim
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(normalized) # [B x 100 x 3] => [B x 100 x 3 x 1]
		# 2D Conv
		x = tf.keras.layers.Conv2D(filters=self.kernel_n, kernel_size=self.params.kernel_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
		# Squeeze
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x)  # remove width axis for 1D Conv [ B x int(100-kernel_width/2) x 1 x kernel_n ] -> [ B x int(100-kernel_width/2) x kernel_n ]
		# 1D Conv * 2
		self.kernel_n += 4
		x = tf.keras.layers.Conv1D(filters=self.kernel_n, kernel_size=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)(x) # [ B x 96 x 10 ]
		self.kernel_n += 4
		x = tf.keras.layers.Conv1D(filters=self.kernel_n, kernel_size=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)(x) # [ B x 94 x 14 ]
		# Pool
		x = tf.keras.layers.AveragePooling1D()(x) # [ B x 47 x 14 ]
		# shape info needed to build decoder model
		self.shape_convolved = x.get_shape().as_list()
		# Flatten
		x = tf.keras.layers.Flatten()(x) #[B x 658]
		# Dense * 3
		x = tf.keras.layers.Dense(int(self.params.z_sz*17), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)  # reduce convolution output
		x = tf.keras.layers.Dense(int(self.params.z_sz*4), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)  # reduce again
		# x = Dense(8, activation=self.params.activation, kernel_initializer=self.params.initializer)(x)

		# *****************************
		#         latent space
		# generate latent vector Q(z|X)

		self.z_mean = tf.keras.layers.Dense(self.params.z_sz, name='z_mean')(x)
		self.z_log_var = tf.keras.layers.Dense(self.params.z_sz, name='z_log_var')(x)

		# use reparameterization trick to push the sampling out as input
		self.z = layers.Sampling()((self.z_mean, self.z_log_var))

		# instantiate encoder model
		encoder = tf.keras.Model(inputs, [self.z, self.z_mean, self.z_log_var], name='encoder')
		encoder.summary()
		# plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
		return encoder

	def build_decoder(self, mean, stdev):
		latent_inputs = tf.keras.layers.Input(shape=(self.params.z_sz,), name='z_sampling')
		# Dense * 3
		x = tf.keras.layers.Dense(int(self.params.z_sz*4), activation=self.params.activation, kernel_initializer=self.params.initializer)(latent_inputs)  # inflate to input-shape/200
		x = tf.keras.layers.Dense(int(self.params.z_sz*17), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)  # double size
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
		# Reshape
		x = tf.keras.layers.Reshape(tuple(self.shape_convolved[1:]))(x)
		# Upsample
		x = tf.keras.layers.UpSampling1D()(x) # [ B x 94 x 16 ]
		# 1D Conv Transpose * 2
		self.kernel_n -= 4
		x = layers.Conv1DTranspose(filters=self.kernel_n, kernel_sz=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)(x) # [ B x 94 x 16 ] -> [ B x 96 x 8 ]
		self.kernel_n -= 4
		x = layers.Conv1DTranspose(filters=self.kernel_n, kernel_sz=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)(x) # [ B x 96 x 8 ] -> [ B x 98 x 4 ]
		# Expand
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2))(x) #  [ B x 98 x 1 x 4 ]
		# 2D Conv Transpose
		x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.params.kernel_sz, name='conv_2d_transpose')(x)
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x) # [B x 100 x 3 x 1] -> [B x 100 x 3]
		outputs_decoder = layers.StdUnnormalization(mean_x=mean, std_x=stdev)(x)

		# instantiate decoder model
		decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
		decoder.summary()
		# plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
		return decoder


	@classmethod
	def load(cls, path):
		custom_objects = {'Sampling': layers.Sampling, 'Conv1DTranspose': layers.Conv1DTranspose, 'StdNormalization': layers.StdNormalization, 'StdUnnormalization': layers.StdUnnormalization}
		return super().load(path=path, custom_objects=custom_objects)
