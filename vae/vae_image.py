import tensorflow as tf

import vae.vae_base as vbase


class VAEimage(vbase.VAE):

	# ***********************************
    #               encoder
    # ***********************************
    def build_encoder(self, inputs):

        x = tf.keras.layers.Dropout(0.5)(inputs)

        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(x) # [B x N_pix x N_pix] => [B x N_pix x N_pix x 1]

        for i in range(3):
            x = tf.keras.layers.Conv2D(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)
            self.filter_n += 4

        x = tf.keras.layers.AveragePooling2D()(x)
        # x = MaxPooling2D( )( x )

        # shape info needed to build decoder model
        self.shape_convolved = x.get_shape().as_list()

        # 3 dense layers
        x = tf.keras.layers.Flatten()(x)
        self.size_convolved = x.get_shape().as_list()
        x = tf.keras.layers.Dense(self.size_convolved[1] // 17, activation='relu',kernel_regularizer=self.regularizer)(x)  # reduce convolution output
        x = tf.keras.layers.Dense(self.size_convolved[1] // 42, activation='relu',kernel_regularizer=self.regularizer)(x)  # reduce again
        #x = Dense(8, activation='relu')(x)

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
        x = tf.keras.layers.Dense(self.size_convolved[1] // 42, activation='relu',kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
        x = tf.keras.layers.Dense(self.size_convolved[1] // 17, activation='relu',kernel_regularizer=self.regularizer)(x)  # double size
        x = tf.keras.layers.Dense(self.shape_convolved[1] * self.shape_convolved[2] * self.shape_convolved[3], activation='relu',kernel_regularizer=self.regularizer)(x)
        x = tf.keras.layers.Reshape((self.shape_convolved[1], self.shape_convolved[2], self.shape_convolved[3]))(x)

        x = tf.keras.layers.UpSampling2D()(x)

        for i in range(3):
            self.filter_n -= 4
            x = tf.keras.layers.Conv2DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu',kernel_regularizer=self.regularizer)(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, padding='same', name='decoder_output')(x)
        outputs_decoder = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x) # [B x N_pix x N_pix x 1] -> [B x N_pix x N_pix]


        # instantiate decoder model
        decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder
