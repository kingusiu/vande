import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
from collections import namedtuple

import vae.losses as losses


# custom sampling layer for latent space
class Sampling(tf.keras.layers.Layer):
	"""Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon

	def get_config(self):
		return super(Sampling, self).get_config()



# custom 1d transposed convolution that expands to 2d output for vae decoder
class Conv1DTranspose(tf.keras.layers.Layer):

	def __init__(self, filters, kernel_sz, activation, **kwargs):
		super(Conv1DTranspose,self).__init__(**kwargs)
		self.kernel_sz = kernel_sz
		self.filters = filters
		self.activation = activation
		self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
		self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(self.kernel_sz,1), activation=self.activation)
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
		config.update({'kernel_sz': self.kernel_sz, 'filters': self.filters, 'activation': self.activation})
		return config


class VAEparticle():

	def __init__(self, input_shape=(100,3), z_sz=10, filter_n=6, kernel_sz=3, loss=losses.make_mse_kl_loss, batch_sz=128, beta=0.01, regularizer=None):
		Parameters = namedtuple('Parameters','input_shape kernel_sz loss regularizer z_sz beta batch_sz')
		self.params = Parameters(input_shape=input_shape, kernel_sz=kernel_sz, loss=loss, regularizer=regularizer, z_sz=z_sz, beta=beta, batch_sz=batch_sz)
		self.filter_n = filter_n

	def build(self, x_mean_var):
		inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
		self.encoder = self.build_encoder(inputs, *x_mean_var)
		self.decoder = self.build_decoder(*x_mean_var)
		outputs = self.decoder(self.z)  # link encoder output to decoder
		# instantiate VAE model
		self.model = tf.keras.Model(inputs, outputs, name='vae')
		self.model.summary()
		self.model.compile(optimizer='adam', loss=self.params.loss(self.z_mean, self.z_log_var, self.params.beta, self.params.batch_sz), experimental_run_tf_function=False)

	def build_encoder(self, inputs, mean, var):
		# normalize
		normalized = tf.keras.layers.Lambda(lambda xx: (xx-mean)/var)(inputs)
		# add channel dim
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(normalized) # [B x 100 x 3] => [B x 100 x 3 x 1]
		# 2D Conv
		x = tf.keras.layers.Conv2D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.params.regularizer)(x)
		# Squeeze
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x)  # remove width axis for 1D Conv [ B x 98 x 1 x filter_n ] -> [ B x 98 x filter_n ]
		# 1D Conv * 2
		self.filter_n += 4
		x = tf.keras.layers.Conv1D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.params.regularizer)(x) # [ B x 96 x 10 ]
		self.filter_n += 4
		x = tf.keras.layers.Conv1D(filters=self.filter_n, kernel_size=self.params.kernel_sz, activation='relu', kernel_regularizer=self.params.regularizer)(x) # [ B x 94 x 14 ]
		# Pool
		x = tf.keras.layers.AveragePooling1D()(x) # [ B x 47 x 14 ]
		# shape info needed to build decoder model
		self.shape_convolved = x.get_shape().as_list()
		# Flatten
		x = tf.keras.layers.Flatten()(x)
		# Dense * 3
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]) // 17, activation='relu', kernel_regularizer=self.params.regularizer)(x)  # reduce convolution output
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]) // 42, activation='relu', kernel_regularizer=self.params.regularizer)(x)  # reduce again
		# x = Dense(8, activation='relu')(x)

		# *****************************
		#         latent space
		# generate latent vector Q(z|X)

		self.z_mean = tf.keras.layers.Dense(self.params.z_sz, name='z_mean', kernel_regularizer=self.params.regularizer)(x)
		self.z_log_var = tf.keras.layers.Dense(self.params.z_sz, name='z_log_var', kernel_regularizer=self.params.regularizer)(x)

		# use reparameterization trick to push the sampling out as input
		self.z = Sampling()((self.z_mean, self.z_log_var))

		# instantiate encoder model
		encoder = tf.keras.Model(inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
		encoder.summary()
		# plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
		return encoder

	def build_decoder(self, mean, var):
		latent_inputs = tf.keras.layers.Input(shape=(self.params.z_sz,), name='z_sampling')
		# Dense * 3
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]) // 42, activation='relu', kernel_regularizer=self.params.regularizer)(latent_inputs)  # inflate to input-shape/200
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]) // 17, activation='relu', kernel_regularizer=self.params.regularizer)(x)  # double size
		x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]), activation='relu', kernel_regularizer=self.params.regularizer)(x)
		# Reshape
		x = tf.keras.layers.Reshape(tuple(self.shape_convolved[1:]))(x)
		# Upsample
		x = tf.keras.layers.UpSampling1D()(x) # [ B x 94 x 16 ]
		# 1D Conv Transpose * 2
		self.filter_n -= 4
		x = Conv1DTranspose(filters=self.filter_n, kernel_sz=self.params.kernel_sz, activation='relu')(x) # [ B x 94 x 16 ] -> [ B x 96 x 8 ]
		self.filter_n -= 4
		x = Conv1DTranspose(filters=self.filter_n, kernel_sz=self.params.kernel_sz, activation='relu')(x) # [ B x 96 x 8 ] -> [ B x 98 x 4 ]
		# Expand
		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2))(x) #  [ B x 98 x 1 x 4 ]
		# 2D Conv Transpose
		x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.params.kernel_sz, activation=tf.keras.activations.elu, kernel_regularizer=self.params.regularizer, name='conv_2d_transpose')(x)
		x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x) # [B x 100 x 3 x 1] -> [B x 100 x 3]
		outputs_decoder = tf.keras.layers.Lambda(lambda xx: (xx*var)+mean, name='un_normalized_decoder_out')(x)

		# instantiate decoder model
		decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
		decoder.summary()
		# plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
		return decoder

	def fit(x_train, x_train, epochs=100, verbose=2):
		self.model.fit(x_train, x_train, epochs=epochs, batch_size=self.batch_sz, verbose=verbose)

	def save(self, path):
		self.model.save(path)

	def load(self, path):
		''' loading only for inference -> passing compile=False '''
		self.model = tf.keras.models.load_model(path, custom_objects={'Sampling': Sampling, 'Conv1DTranspose': Conv1DTranspose}, compile=False)
		

