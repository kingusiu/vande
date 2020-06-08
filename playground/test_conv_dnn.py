import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D


class Conv1DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, activation):
        super(Conv1DTranspose,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def call(self, inputs, **kwargs):
        # expand input and kernel to 2D
        x = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=2))(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        kernel_size = (self.kernel_size,1) # [ 3 ] -> [ 3 x 1 ]
        # call Conv2DTranspose
        x = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=kernel_size, activation=self.activation)(x)
        # squeeze back to 1D and return
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x,axis=2))(x)
        return x


x = np.random.random(size=(1000,100,3,1))
y = np.random.randint(5,size=1000)

# encoder

input = tf.keras.Input(shape=(100,3,1))
x = tf.keras.layers.Conv2D(filters=4,kernel_size=3,activation='relu')(input)
shape = x.get_shape().as_list()
x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x) # remove width axis for 1D Conv [ B x 98 x 1 x 4 ] -> [ B x 98 x 4 ]
shape = x.get_shape().as_list() # [B x 98 x 4]
x = tf.keras.layers.Conv1D(filters=8,kernel_size=3,activation='relu')(x)
shape_convolved = x.get_shape().as_list()
# flatten for dense
x = tf.keras.layers.Flatten()(x)
# Dense
x = tf.keras.layers.Dense(10,activation='relu')(x)
z = tf.keras.layers.Dense(1,activation='relu')(x)

# decoder

x = tf.keras.layers.Dense(10,activation='relu')(z)
x = tf.keras.layers.Dense(shape_convolved[1]*shape_convolved[2],activation='relu')(x)
x = tf.keras.layers.Reshape((shape_convolved[1],shape_convolved[2]))(x) # [B x 96 x 8]
x = Conv1DTranspose(filters=4,kernel_size=3,activation='relu')(x) # [B x 98 x 4]
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2))(x) # [B x 98 x 1 x 4]
output = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='relu')(x)


model = tf.keras.Model(input, output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)