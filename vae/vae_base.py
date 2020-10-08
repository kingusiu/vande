from abc import ABC, abstractmethod
import os
import matplotlib.pyplot as plt
import pathlib
import h5py

import tensorflow as tf
from collections import namedtuple

import config.config as co
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


class VAE(ABC):

    regularizer_dict = {
        'L1' : tf.keras.regularizers.l1(0.01),
        'L2' : tf.keras.regularizers.l2(0.01),
        'L1L2': tf.keras.regularizers.L1L2(l1=0.01,l2=0.01),
    }

    def __init__(self, **params):
        Parameters = namedtuple('Parameters', sorted(params))
        self.params = Parameters(**params)
        self.filter_n = self.params.filter_ini_n
        self.regularizer = self.regularizer_dict[self.params.regularizer] if self.params.regularizer is not None else None

    def build(self, x_mean_stdev):
        # build and link encoder and decoder
        inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
        self.encoder = self.build_encoder(inputs, *x_mean_stdev)
        self.decoder = self.build_decoder(*x_mean_stdev)
        outputs = self.decoder(self.z)  # link encoder output to decoder
        # instantiate VAE model
        self.model = tf.keras.Model(inputs, outputs, name='vae')
        self.model.summary()
        self.model.compile(optimizer='adam', loss=self.params.loss(self.z_mean, self.z_log_var, self.params.beta), metrics=[self.params.reco_loss, losses.make_kl_loss(self.z_mean,self.z_log_var)], experimental_run_tf_function=False)

    @abstractmethod
    def build_encoder(self, inputs):
        pass

    @abstractmethod
    def build_decoder(self):
        pass

    @classmethod
    def from_saved_model(cls, path):
        encoder, decoder, model = cls.load(path)
        with h5py.File(os.path.join(path,'model_params.h5'),'r') as f: 
            params = f.get('params')
            beta = params.attrs['beta']
        instance = cls(beta=beta)
        instance.encoder = encoder
        instance.decoder = decoder
        instance.model = model
        return instance

    @property
    def beta(self):
        return self.params.beta

    def fit( self, x, y, epochs=3, verbose=2 ):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN(),
                     ] #TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.history = self.model.fit(x, y, batch_size=self.params.batch_sz, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=0.25)
        return self.history

    def predict(self, x):
        return self.model.predict(x, batch_size=1024)

    def predict_with_latent(self, x):
        z_mean, z_log_var, z = self.encoder.predict(x, batch_size=1024)
        reco = self.decoder.predict(z, batch_size=1024)
        return [reco, z_mean, z_log_var]

    def save(self, path):
        print('saving model to {}'.format(path))
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path,'decoder.h5'))
        self.model.save(os.path.join(path,'vae.h5'))
        # sneak in beta factor as group attribute of vae.h5 file
        with h5py.File(os.path.join(path,'model_params.h5'),'w') as f:
            ds = f.create_group('params')
            ds.attrs['beta'] = self.params.beta

    @classmethod
    def load(cls, path, custom_objects={}):
        ''' loading only for inference -> passing compile=False '''
        encoder = tf.keras.models.load_model(os.path.join(path,'encoder.h5'), custom_objects=custom_objects, compile=False)
        decoder = tf.keras.models.load_model(os.path.join(path,'decoder.h5'), custom_objects=custom_objects, compile=False)
        model = tf.keras.models.load_model(os.path.join(path,'vae.h5'), custom_objects=custom_objects, compile=False)
        return encoder, decoder, model

    def plot_training(self, fig_dir='fig'):
        plt.figure()
        plt.semilogy(self.history.history['loss'])
        plt.semilogy(self.history.history['val_loss'])
        plt.title('training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training','validation'], loc='upper right')
        plt.savefig(os.path.join(fig_dir,'loss.png'))
        plt.close()

    def sample_pixels_from_dist(self,dist):
        return np.random.exponential(1. / dist)  # numpy exponential dist takes 1/k param instead of k param
