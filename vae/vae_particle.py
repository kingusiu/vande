import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
from collections import namedtuple
import matplotlib.pyplot as plt

import vae.losses as losses
import vae.vae_model as baseVAE


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


class VAEparticle(baseVAE.VAE):

	def __init__(self, input_shape=(100,3), z_sz=10, filter_ini_n=6, kernel_sz=3, loss=losses.make_threeD_kl_loss, reco_loss=losses.threeD_loss, batch_sz=128, beta=0.01, regularizer=None):
		super(VAEparticle, self).__init__(input_shape=input_shape, z_sz=z_sz, filter_ini_n=filter_ini_n, kernel_sz=kernel_sz, loss=loss, reco_loss=reco_loss, regularizer=regularizer, beta=beta, batch_sz=batch_sz)

	def build(self, x_mean_stdev):
		inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
		self.encoder = self.build_encoder(inputs, *x_mean_stdev)
		self.decoder = self.build_decoder(*x_mean_stdev)
		outputs = self.decoder(self.z)  # link encoder output to decoder
		# instantiate VAE model
		self.model = tf.keras.Model(inputs, outputs, name='vae')
		self.model.summary()
		self.model.compile(optimizer='adam', loss=self.params.loss(self.z_mean, self.z_log_var, self.params.beta), metrics=[self.params.reco_loss, losses.make_kl_loss(self.z_mean,self.z_log_var)], experimental_run_tf_function=False)

	def build_encoder(self, inputs, mean, stdev):
		# normalize
		normalized = tf.keras.layers.Lambda(lambda xx: (xx-mean)/stdev)(inputs)
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

	def build_decoder(self, mean, stdev):
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
		outputs_decoder = tf.keras.layers.Lambda(lambda xx: (xx*stdev)+mean, name='un_normalized_decoder_out')(x)

		# instantiate decoder model
		decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
		decoder.summary()
		# plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
		return decoder

	def fit(self, x_train, epochs=100, verbose=2, reco_loss=losses.threeD_loss):
		self.history = self.model.fit(x_train, x_train, epochs=epochs, batch_size=self.params.batch_sz, verbose=verbose, validation_split=0.25)

	def save(self, path):
		print('saving model to {}'.format(path))
		self.encoder.save(os.path.join(path, 'encoder.h5'))
		self.decoder.save(os.path.join(path,'decoder.h5'))
		self.model.save(os.path.join(path,'vae.h5'))

	def load(self, path):
		''' loading only for inference -> passing compile=False '''
		custom_objects = {'Sampling': Sampling, 'Conv1DTranspose': Conv1DTranspose}
		self.encoder = tf.keras.models.load_model(os.path.join(path,'encoder.h5'), custom_objects=custom_objects, compile=False)
		self.decoder = tf.keras.models.load_model(os.path.join(path,'decoder.h5'), custom_objects=custom_objects, compile=False)
		self.model = tf.keras.models.load_model(os.path.join(path,'vae.h5'), custom_objects=custom_objects, compile=False)

	def plot_training(self, fig_dir='fig' ):
		plt.figure()
		plt.semilogy(self.history.history['loss'])
		plt.semilogy(self.history.history['val_loss'])
		plt.title('training and validation loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training','validation'], loc='upper right')
		plt.savefig(os.path.join(fig_dir,'loss.png'))
		plt.close()

