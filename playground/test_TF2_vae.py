import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
print('tensorflow version: ', tf.__version__)

import vae.losses as lo
import vae.vae_particle as vap


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


original_dim = 5
latent_dim = 3
intermediate_dim = 4
batch_sz = 10

class VAE():

    def __init__(self, input_shape=(original_dim,)):
        self.input_shape = input_shape

    def build(self, loss):
        inputs = tf.keras.layers.Input(shape=self.input_shape, dtype=tf.float32, name='encoder_input')
        self.encoder = self.build_encoder(inputs)
        self.decoder = self.build_decoder()
        outputs = self.decoder(self.z)  # link encoder output to decoder
        # instantiate VAE model
        self.model = tf.keras.Model(inputs, outputs, name='vae')
        self.model.summary()
        self.model.compile(optimizer='adam', loss=loss(batch_sz))

    def build_encoder(self, inputs):
        # Define encoder model.
        x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(inputs)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        self.z = Sampling()((self.z_mean, self.z_log_var))
        return tf.keras.Model(inputs=inputs, outputs=self.z, name="encoder")

    def build_decoder(self):
        # Define decoder model.
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
        x = tf.keras.layers.Dense(intermediate_dim, name='decode_dense_1', activation="relu")(latent_inputs)
        outputs = tf.keras.layers.Dense(original_dim, name='decode_out',activation="sigmoid")(x)
        return tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        ''' loading only for inference -> passing compile=False '''
        self.model = tf.keras.models.load_model(path, custom_objects={'Sampling': Sampling}, compile=False)
        print('loaded model ', self.model)


import numpy as np
examples_n = 300
#x_train = np.random.random(size=(examples_n,original_dim))
x_train = np.random.random(size=(examples_n,100,3))
nn = vap.VAEparticle(loss=lo.make_threeD_kl_loss)
x_mean_var = (np.mean(x_train), np.var(x_train))
nn.build(x_mean_var=x_mean_var)
nn.fit(x_train, epochs=3, verbose=2)
x_test = np.random.random(size=(examples_n,100,3))
x_predicted = nn.model.predict(x_test)
path = './test_model.h5'
nn.save(path)
vae_loaded = vap.VAEparticle()
vae_loaded.load(path)
x_predicted_loaded = vae_loaded.model.predict(x_test)
print('x_predicted')
print(x_predicted)
print('x_loaded')
print(x_predicted_loaded)
#assert np.allclose(x_predicted, x_predicted_loaded) => can not compare 2 predictions because of probabilistic sampling layer in between
weights = nn.model.get_weights()
weights_loaded = vae_loaded.model.get_weights()
for w1, w2 in zip(weights, weights_loaded):
    assert np.allclose(w1, w2) 
print('-'*10, 'finished run', '-'*10)
exit()

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = tf.keras.layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)


vae.save('test_vae_model.h5')