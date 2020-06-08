import tensorflow as tf

from keras.layers import Input, Dense, Lambda, Flatten, Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, Reshape, Conv2DTranspose, UpSampling1D, Layer
from keras.models import Model, load_model

from vae.vae_model import VAE
from vae.losses import *

# custom 1d transposed convolution that expands to 2d output for vae decoder
class Conv1DTranspose(Layer):

    def __init__(self, filters, kernel_size, activation):
        super(Conv1DTranspose,self).__init__()
        self.kernel_size = (kernel_size,1) # [ 3 ] -> [ 3 x 1 ]
        self.ExpandChannel = Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = Conv2DTranspose(filters=filters, kernel_size=self.kernel_size, activation=activation)
        self.SqueezeChannel = Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs, **kwargs):
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config



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
        x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)  # remove width axis for 1D Conv [ B x 98 x 1 x filter_n ] -> [ B x 98 x filter_n ]
        self.filter_n += 4
        x = Conv1D(filters=self.filter_n,kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x) # [ B x 96 x 10 ]
        self.filter_n += 4
        x = Conv1D(filters=self.filter_n,kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x) # [ B x 94 x 14 ]

        x = AveragePooling1D()(x) # [ B x 47 x 14 ]

        # shape info needed to build decoder model
        self.shape_convolved = x.get_shape().as_list()

        # 3 dense layers
        x = Flatten()(x)
        x = Dense(np.prod(self.shape_convolved[1:]) // 17, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce convolution output
        x = Dense(np.prod(self.shape_convolved[1:]) // 42, activation='relu', kernel_regularizer=self.regularizer)(x)  # reduce again
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
        x = Dense(np.prod(self.shape_convolved[1:]) // 42, activation='relu', kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
        x = Dense(np.prod(self.shape_convolved[1:]) // 17, activation='relu', kernel_regularizer=self.regularizer)(x)  # double size
        x = Dense(np.prod(self.shape_convolved[1:]), activation='relu', kernel_regularizer=self.regularizer)(x)
        x = Reshape(tuple(self.shape_convolved[1:]))(x)

        x = UpSampling1D()(x) # [ B x 94 x 16 ]

        self.filter_n -= 4
        x = Conv1DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu')(x) # [ B x 94 x 16 ] -> [ B x 96 x 8 ]
        self.filter_n -= 4
        x = Conv1DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu')(x) # [ B x 96 x 8 ] -> [ B x 98 x 4 ]

        x = Lambda(lambda x: tf.expand_dims(x,axis=2))(x) #  [ B x 98 x 1 x 4 ]
        outputs_decoder = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, name='decoder_output')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs_decoder, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder