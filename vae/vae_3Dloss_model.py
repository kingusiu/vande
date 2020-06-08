import tensorflow as tf

from keras.layers import Input, Dense, Lambda, Flatten, Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, Reshape, Conv2DTranspose, UpSampling1D
from keras.models import Model, load_model

from vae.vae_model import VAE
from vae.losses import *

class VAE_3D( VAE ):

    def __init__(self,**kwargs):
        super(VAE_3D,self).__init__(**kwargs)
        self.input_shape = (100,3,1)

    def load( self, run = 0 ):
        pass
        # todo: load with custom 3d loss name

    def compile(self,model):
        model.compile(optimizer='adam', loss=threeD_kl_loss(self.z_mean, self.z_log_var), metrics=[threeD_loss,kl_loss_for_metric(self.z_mean,self.z_log_var)])  # , metrics=loss_metrics monitor mse and kl terms of loss 'rmsprop'

    # ***********************************
    #               encoder
    # ***********************************
    def build_encoder(self, inputs):
        x = inputs
        x = Conv2D(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)
        x = tf.squeeze(x, axis=2)  # remove width axis for 1D Conv [ B x 98 x 1 x filter_n ] -> [ B x 98 x filter_n ]
        self.filter_n += 4
        x = Conv1D(filters=self.filter_n,kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)
        self.filter_n += 4
        x = Conv1D(filters=self.filter_n,kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)

        x = AveragePooling1D()(x)

        # shape info needed to build decoder model
        self.shape_convolved = K.int_shape(x)

        # 3 dense layers
        x = Flatten()(x)
        self.size_convolved = K.int_shape(x)
        x = Dense(self.size_convolved[1] // 17, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce convolution output
        x = Dense(self.size_convolved[1] // 42, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce again
        # x = Dense(8, activation='relu')(x)

        # *****************************
        #         latent space
        # generate latent vector Q(z|X)

        self.z_mean = Dense(self.z_size, name='z_mean', kernel_regularizer=self.regularizer)(x)
        self.z_log_var = Dense(self.z_size, name='z_log_var', kernel_regularizer=self.regularizer)(x)

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
        x = Dense(self.size_convolved[1] // 42, activation='relu', kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
        x = Dense(self.size_convolved[1] // 17, activation='relu', kernel_regularizer=self.regularizer)(x)  # double size
        x = Dense(self.shape_convolved[1] * self.shape_convolved[2] * self.shape_convolved[3], activation='relu', kernel_regularizer=self.regularizer)(x)
        x = Reshape((self.shape_convolved[1], self.shape_convolved[2], self.shape_convolved[3]))(x)

        x = UpSampling1D()(x)

        # todo: implement conv1transpose
        x = Conv1DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu',kernel_regularizer=self.regularizer, padding='same')(x)

        # todo: change to conv1dT, conv2dT
        for i in range(3):
            self.filter_n -= 4
            x = Conv2DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, padding='same')(x)

        outputs_decoder = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, padding='same', name='decoder_output')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs_decoder, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder