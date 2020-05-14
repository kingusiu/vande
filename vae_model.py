import os
from keras.layers import Input, Dense, Lambda, Flatten, Conv2D, AveragePooling2D, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
import keras.losses
import keras.backend as K
from config import *
from losses import *


class VAE( object ):

    def __init__(self):
        # network parameters
        self.input_shape = (config['image_size'], config['image_size'], 1)
        self.batch_size = 128
        self.kernel_size = 3
        self.filter_n = 4
        self.z_size = 5
        self.encoder = None
        self.decoder = None
        self.model = None


    # adding keras mse as dummy loss, because training loss in function closure not (easily) accessible and model won't load without all custom function references
    def load( self, run = 0 ):

        model_dir = "./models"
        self.encoder = load_model(os.path.join(model_dir, 'encoder_run_' + str(run) + '.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss, 'sampling' : self.sampling})
        self.decoder = load_model(os.path.join(model_dir, 'decoder_run_' + str(run) + '.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss})
        self.model = load_model(os.path.join(model_dir, 'vae_run_' + str(run) + '.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss, 'loss': keras.losses.mse, 'sampling' : self.sampling})


    def build( self ):

        inputs = Input(shape=self.input_shape, name='encoder_input')
        self.encoder = self.build_encoder( inputs )
        self.decoder = self.build_decoder( )
        outputs = self.decoder(self.encoder(inputs)[-1])  # link encoder output to decoder
        # instantiate VAE model
        vae = Model(inputs, outputs, name='vae')
        vae.summary()
        vae.compile(optimizer='adam', loss=mse_kl_loss(self.z_mean,self.z_log_var), metrics=[mse_loss,kl_loss_for_metric(self.z_mean,self.z_log_var)])  # , metrics=loss_metrics monitor mse and kl terms of loss 'rmsprop'
        self.model = vae


    # ***********************************
    #               encoder
    # ***********************************
    def build_encoder(self, inputs):

        x = inputs
        for i in range(3):
            x = Conv2D(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu')(x)
            self.filter_n += 4

        x = AveragePooling2D()(x)
        # x = MaxPooling2D( )( x )

        # shape info needed to build decoder model
        self.shape_convolved = K.int_shape(x)

        # 3 dense layers
        x = Flatten()(x)
        self.size_convolved = K.int_shape(x)
        x = Dense(self.size_convolved[1] // 100, activation='relu')(x)  # reduce convolution output
        x = Dense(self.size_convolved[1] // 200, activation='relu')(x)  # reduce again
        x = Dense(8, activation='relu')(x)

        # *****************************
        #           latent space

        # generate latent vector Q(z|X)
        self.z_mean = Dense(self.z_size, name='z_mean')(x)
        self.z_log_var = Dense(self.z_size, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self.z_size,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        encoder.summary()
        # plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
        return encoder


    # ***********************************
    #           decoder
    # ***********************************
    def build_decoder(self):

        latent_inputs = Input(shape=(self.z_size,), name='z_sampling')
        x = Dense(self.size_convolved[1] // 200, activation='relu')(latent_inputs)  # inflate to input-shape/200
        x = Dense(self.size_convolved[1] // 100, activation='relu')(x)  # double size
        x = Dense(self.shape_convolved[1] * self.shape_convolved[2] * self.shape_convolved[3], activation='relu')(x)
        x = Reshape((self.shape_convolved[1], self.shape_convolved[2], self.shape_convolved[3]))(x)

        x = UpSampling2D()(x)

        for i in range(3):
            self.filter_n -= 4
            x = Conv2DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu')(x)

        outputs_decoder = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu',
                                          padding='same', name='decoder_output')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs_decoder, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder


    def fit( self, x, y, epochs=3, verbose=2 ):
        callbacks = [EarlyStopping(monitor='val_loss', patience=7, verbose=1),ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),TerminateOnNaN()]
        return self.model.fit( x, y, batch_size=self.batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks )


    # ***********************************
    #       reparametrization trick
    # ***********************************
    def sampling( self, args ):
        """
        instead of sampling from Q(z|X),
        sample eps = N(0,I), then z = z_mean + sqrt(var)*eps

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + K.sqrt(K.exp(z_log_var)) * epsilon


    def save_model( self, run = 0 ):
        self.encoder.save('models/encoder_run_' + str(run) + '.h5')
        self.decoder.save('models/decoder_run_' + str(run) + '.h5')
        self.model.save('models/vae_run_' + str(run) + '.h5')

