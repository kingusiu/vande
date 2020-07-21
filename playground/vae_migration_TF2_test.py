import tensorflow as tf
import util as ut

class VAE(tf.keras.Model):

	def build(self):
		pass

	def encoder(self):
		pass

	def decoder(self):
		pass


# data
dat, y = ut.get_test_data_for_vae(100,7)

vae = VAE()
#vae.compile()
#vae.fit()

input = tf.keras.Input(shape=dat.shape[1:])
x = tf.keras.layers.Dense(10)(input)
x = tf.keras.layers.Lambda(lambda x: x * 2 + 1)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(input,output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)
model.fit(dat,y,batch_size=10,epochs=3,verbose=2)

model.save('test_model.h5')