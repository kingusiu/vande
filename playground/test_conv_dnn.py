import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D


def sampling(args):

    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    # by default, random_normal has mean=0 and std=1.0
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Conv1DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, activation):
        super(Conv1DTranspose,self).__init__()
        self.kernel_size = (kernel_size,1) # [ 3 ] -> [ 3 x 1 ]
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=self.kernel_size, activation=activation)
        self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

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



class Conv1Dto2DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, activation):
        super(Conv1Dto2DTranspose,self).__init__()
        self.kernel_size = (kernel_size,1) # [ 3 ] -> [ 3 x 1 ]
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=self.kernel_size, activation=activation)
        #self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs, **kwargs):
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        #x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        config = super(Conv1Dto2DTranspose, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


samples = np.random.random(size=(1000,100,3,1))
y = np.random.randint(5,size=1000)

# encoder

input = tf.keras.Input(shape=(100,3,1))
# 2D Conv
x = tf.keras.layers.Conv2D(filters=4,kernel_size=3,activation='relu')(input)
# Squeeze
x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x) # remove width axis for 1D Conv [ B x 98 x 1 x 4 ] -> [ B x 98 x 4 ]
# 1D Conv * 2
x = tf.keras.layers.Conv1D(filters=8,kernel_size=3,activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=12,kernel_size=3,activation='relu')(x)
# Pool
x = tf.keras.layers.AveragePooling1D()(x)  # [ B x 47 x 14 ]
shape_convolved = x.get_shape().as_list() # [B x 94 x 12]
# Flatten
x = tf.keras.layers.Flatten()(x)
# Dense * 2
x = tf.keras.layers.Dense(20,activation='relu')(x)
x = tf.keras.layers.Dense(10,activation='relu')(x)
# latent space
z_mean = tf.keras.layers.Dense(5)(x)
z_var = tf.keras.layers.Dense(5)(x)
z = tf.keras.layers.Lambda( sampling )([z_mean,z_var])

# decoder

# Dense * 3
x = tf.keras.layers.Dense(10,activation='relu')(z)
x = tf.keras.layers.Dense(20,activation='relu')(x)
x = tf.keras.layers.Dense(np.prod(shape_convolved[1:]),activation='relu')(x)
# Reshape
x = tf.keras.layers.Reshape(tuple(shape_convolved[1:]))(x) # [B x 94 x 12]
# Upsample
x = tf.keras.layers.UpSampling1D()(x) # [ B x 94 x 16 ]
# 1D Conv Transpose * 2
x = Conv1DTranspose(filters=8,kernel_size=3,activation='relu')(x) # [B x 94 x 12] -> [B x 96 x 1 x 8]
x = Conv1DTranspose(filters=4,kernel_size=3,activation='relu')(x) # [B x 96 x 8] -> [B x 98 x 1 x 4]
# Expand
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2))(x)
# 2D Conv Transpose
output = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='relu')(x)


model = tf.keras.Model(input, output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)
model.fit(samples,samples,batch_size=10,epochs=12,verbose=2)

model.save('../models/test_conv_vae.h5')