from abc import ABC, abstractmethod
import os
import matplotlib.pyplot as plt
import pathlib

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

    def __init__(self, **params):
        Parameters = namedtuple('Parameters', sorted(params))
        self.params = Parameters(**params)
        self.filter_n = self.params.filter_ini_n



    def build(self, x_mean_stdev):
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


    def fit( self, x, y, epochs=3, verbose=2 ):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN(),
                     ] #TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.history = self.model.fit(x, y, batch_size=self.batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=0.25)
        return self.history


    def predict(self, x):
        return self.model.predict( x, batch_size=self.batch_size )


    def predict_with_latent(self, x):
        z_mean, z_log_var, z = self.encoder.predict(x, batch_size=self.batch_size)
        reco = self.decoder.predict(z, batch_size=self.batch_size)
        return [ reco, z_mean, z_log_var ]


    def save_model(self):
        print('saving model to {}'.format(self.model_dir))
        self.encoder.save(os.path.join(self.model_dir, 'encoder.h5'))
        self.decoder.save(os.path.join(self.model_dir,'decoder.h5'))
        self.model.save(os.path.join(self.model_dir,'vae.h5'))

    def load( self ):
        self.encoder = tf.keras.models.load_model(os.path.join(self.model_dir, 'encoder.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss_fun': tf.keras.losses.mse, 'kl_loss_for_metric': kl_loss_for_metric, 'Sampling' : Sampling})
        self.decoder = tf.keras.models.load_model(os.path.join(self.model_dir, 'decoder.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss_fun': tf.keras.losses.mse, 'kl_loss_for_metric': kl_loss_for_metric})
        self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'vae.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss_fun': tf.keras.losses.mse, 'kl_loss_for_metric': kl_loss_for_metric, 'loss': tf.keras.losses.mse, 'kl_loss_fun': kl_loss, 'Sampling' : Sampling})

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
