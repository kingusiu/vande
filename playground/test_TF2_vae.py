import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

original_dim = 7
latent_dim = 3
intermediate_dim = 5

class VAE():

    def build_encoder(self):
        # Define encoder model.
        self.original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
        x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(self.original_inputs)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        self.z = Sampling()((self.z_mean, self.z_log_var))
        return tf.keras.Model(inputs=self.original_inputs, outputs=self.z, name="encoder")

    def build_decoder(self):
        # Define decoder model.
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
        x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
        self.outputs = tf.keras.layers.Dense(original_dim, activation="sigmoid")(x)
        return tf.keras.Model(inputs=latent_inputs, outputs=self.outputs, name="decoder")        

    def run(self,x_train):
        
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        

        # Define VAE model.
        self.outputs = decoder(self.z)
        vae = tf.keras.Model(inputs=self.original_inputs, outputs=self.outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var) + 1)
        vae.add_loss(kl_loss)

        # Train.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        vae.fit(x_train, x_train, epochs=3, batch_size=64)
        vae.save('test_vae_model.h5')



import numpy as np
x_train = np.random.random(size=(100,original_dim))

vae = VAE()
vae.run(x_train)
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