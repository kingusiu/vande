import tensorflow as tf

from vae.vae_model import VAE
from vae.vae_model import Sampling


class VAE_HR(VAE):

	def __init__(self, *args, **kwargs):
		super(VAE_HR, self).__init__(*args, **kwargs)
		self.dense_sz = 128

	# ***********************************
	#               encoder
	# ***********************************

	def build_encoder(self, inputs):

		x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(inputs) # [B x N_pix x N_pix] => [B x N_pix x N_pix x 1]

		for i in range(3):
			x = tf.keras.layers.Dropout(0.1)(x)
			x = tf.keras.layers.Conv2D(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)
			x = tf.keras.layers.AveragePooling2D()(x)

		# shape info needed to build decoder model
		self.shape_convolved = x.get_shape().as_list()

		# 3 dense layers
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(self.dense_sz, activation='relu',kernel_regularizer=self.regularizer)(x)  # reduce convolution output

		# *****************************
		#         latent space
		# generate latent vector Q(z|X)

		self.z_mean = tf.keras.layers.Dense(self.z_size, name='z_mean', kernel_regularizer=self.regularizer)(x)
		self.z_log_var = tf.keras.layers.Dense(self.z_size, name='z_log_var', kernel_regularizer=self.regularizer)(x)

		# use reparameterization trick to push the sampling out as input
		self.z = Sampling()((self.z_mean, self.z_log_var))

		# instantiate encoder model
		encoder = tf.keras.Model(inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
		encoder.summary()
		# plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
		return encoder


	# ***********************************
	#           decoder
	# ***********************************
	def build_decoder(self):

		latent_inputs = tf.keras.layers.Input(shape=(self.z_size,), name='z_sampling')
		x = tf.keras.layers.Dense(self.dense_sz, activation='relu',kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
		x = tf.keras.layers.Dense(self.shape_convolved[1] * self.shape_convolved[2] * self.shape_convolved[3], activation='relu',kernel_regularizer=self.regularizer)(x)
		x = tf.keras.layers.Reshape((self.shape_convolved[1], self.shape_convolved[2], self.shape_convolved[3]))(x)

		for i in range(3):
			x = tf.keras.layers.UpSampling2D()(x)
			x = tf.keras.layers.Conv2DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu',kernel_regularizer=self.regularizer)(x)
			x = tf.keras.layers.Dropout(0.1)(x)

		# last conv transpose to get back to 1 channel
		x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, padding='same', name='decoder_output')(x)

		outputs_decoder = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x) # [B x N_pix x N_pix x 1] -> [B x N_pix x N_pix]


		# instantiate decoder model
		decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
		decoder.summary()
		# plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
		return decoder
