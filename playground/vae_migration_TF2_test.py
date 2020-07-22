import tensorflow as tf
import util as ut

class CustomLoss(tf.keras.losses.Loss):

	def __init__(self, extra_param, name='CustomLoss'):
		super(CustomLoss,self).__init__(name=name)
		self.extra_param = extra_param

	def extra_loss(self, inputs, outputs):
		return tf.math.reduce_mean(self.extra_param)

	def call(self, inputs, outputs):
		return tf.math.reduce_mean(inputs-outputs) + self.extra_loss(inputs, outputs)


class CustomLayer(tf.keras.layers.Layer):

	def __init__(self, **kwargs):
		super(CustomLayer,self).__init__(**kwargs)
		self.layer = tf.keras.layers.Dense(5)

	def call(self, inputs):
		return self.layer(inputs)

	def get_config(self):
		return super(CustomLayer, self).get_config()


class VAE():

	def __init__(self):
		self.norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()

	def build(self):
		input = tf.keras.Input(shape=dat.shape[1:])
		x = self.norm_layer(input)
		self.var, self.mean = tf.keras.layers.Lambda(lambda xx: (xx[0],xx[1]))((self.norm_layer.variance, self.norm_layer.mean))
		output = tf.keras.layers.Lambda(lambda xx, vv=self.var, mm=self.mean: xx * tf.sqrt(vv) + mm)(x)

		self.model = tf.keras.Model(input,output)
		self.model.summary()
		self.model.compile(optimizer='adam',loss=tf.keras.losses.mse, experimental_run_tf_function=False)
		
	def run(self, dat):
		self.norm_layer.adapt(dat)
		self.model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)
		self.model.save('test_model.h5')



import sys
print(sys.version)
print(tf.__version__)

# data
#dat, y = ut.get_test_data_for_vae(100,7)
import numpy as np
dat = np.random.random(size=(100,7))

#vae = VAE()
#vae.build()
#vae.run(dat)
#exit()

########## 1: global scope ##########

input = tf.keras.Input(shape=dat.shape[1:])
norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
x = norm_layer(input)
output = tf.keras.layers.Lambda(lambda xx: xx * tf.sqrt(norm_layer.variance) + norm_layer.mean)(x)

model = tf.keras.Model(input,output)
model.compile(optimizer='adam',loss=tf.keras.losses.mse)
norm_layer.adapt(dat)
model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)

model.save('test_model.h5')
print('\n', '*'*20, 'global scope: model saved', '*'*20)

########## 2: function scope ##########

def function_scope(dat):
	input = tf.keras.Input(shape=dat.shape[1:])
	norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
	x = norm_layer(input)
	output = tf.keras.layers.Lambda(lambda xx: xx * tf.sqrt(norm_layer.variance) + norm_layer.mean)(x)

	model = tf.keras.Model(input,output)
	model.compile(optimizer='adam',loss=tf.keras.losses.mse, experimental_run_tf_function=False)
	norm_layer.adapt(dat)
	model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)

	model.save('test_model.h5')
	print('*'*20, 'function scope: model saved', '*'*20)


#function_scope(dat)

########## 3: function scope with lambda wrap ##########

def function_scope_lambda_wrap(dat):
	input = tf.keras.Input(shape=dat.shape[1:])
	norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
	x = norm_layer(input)
	var, mean = tf.keras.layers.Lambda(lambda xx: (xx[0],xx[1]))((norm_layer.variance, norm_layer.mean))
	output = tf.keras.layers.Lambda(lambda xx: xx * tf.sqrt(var) + mean)(x)

	model = tf.keras.Model(input,output)
	model.summary()
	model.compile(optimizer='adam',loss=tf.keras.losses.mse, experimental_run_tf_function=False)
	norm_layer.adapt(dat)
	model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)

	model.save('test_model.h5')
	print('*'*20, 'function scope lambda wrap: model saved', '*'*20)


#function_scope_lambda_wrap(dat)

class UnNormalization(tf.keras.layers.Layer):

	def __init__(self, norm_mean, norm_var, **kwargs):
		super(UnNormalization, self).__init__(**kwargs)
		self.norm_mean = norm_mean
		self.norm_var = norm_var
		#def unnnorm_closure(xx):
		#	return xx * tf.sqrt(self.norm_var) + self.norm_mean
		self.unnormalize = tf.keras.layers.Lambda(lambda xx: xx * tf.sqrt(self.norm_var) + self.norm_mean)

	def call(self, inputs):
		return self.unnormalize(inputs)

	def get_config(self):
		config = super(UnNormalization, self).get_config()
		config.update({'norm_mean': self.norm_mean, 'norm_var': self.norm_var})
		return config


def function_scope_custom_custom_layer(dat):
	input = tf.keras.Input(shape=dat.shape[1:])
	norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
	x = norm_layer(input)
	output = UnNormalization(norm_layer.mean, norm_layer.variance)(x)

	model = tf.keras.Model(input, output)
	model.summary()
	model.compile(optimizer='adam', loss=tf.keras.losses.mse, experimental_run_tf_function=False)
	norm_layer.adapt(dat)
	model.fit(dat, dat, batch_size=10, epochs=3, verbose=2)

	model.save('test_model.h5')
	print('*'*20, 'function scope custom layer: model saved', '*'*20)


#function_scope_custom_custom_layer(dat)


def function_scope_closure(dat):
	input = tf.keras.Input(shape=dat.shape[1:])
	norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
	x = norm_layer(input)
	def unnnorm_closure(xx):
			return xx * tf.sqrt(norm_layer.variance) + norm_layer.mean
	output = tf.keras.layers.Lambda(lambda xx: unnnorm_closure(xx))(x)

	model = tf.keras.Model(input, output)
	model.summary()
	model.compile(optimizer='adam', loss=tf.keras.losses.mse)
	norm_layer.adapt(dat)
	model.fit(dat, dat, batch_size=10, epochs=3, verbose=2)

	model.save('test_model.h5')
	print('*'*20, 'function scope closure: model saved', '*'*20)


function_scope_closure(dat)
exit()

def outside_of_class_minimal():
	input = tf.keras.Input(shape=dat.shape[1:])
	norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
	x = norm_layer(input)
	var, mean = tf.keras.layers.Lambda(lambda xx: (xx[0],xx[1]))((norm_layer.variance, norm_layer.mean))
	output = tf.keras.layers.Lambda(lambda xx: xx * tf.sqrt(var) + mean)(x)

	model = tf.keras.Model(input,output)
	model.summary()
	norm_layer.adapt(dat)
	model.compile(optimizer='adam',loss=tf.keras.losses.mse, experimental_run_tf_function=False)
	model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)

	model.save('test_model.h5')


def inside_function_2():

	input = tf.keras.Input(shape=dat.shape[1:])
	in_eval = tf.keras.layers.Lambda(lambda xx: tf.math.reduce_mean(xx))(input)
	def closure_f(x):
		return x + tf.math.reduce_mean(input)
	#output = tf.keras.layers.Lambda(lambda xx: closure_f(xx))(input)
	output = tf.keras.layers.Lambda(lambda xx: xx + in_eval)(input)

	model = tf.keras.Model(input,output)
	model.summary()
	model.compile(optimizer='adam',loss=tf.keras.losses.mse, experimental_run_tf_function=False)
	model.fit(dat,dat,batch_size=10,epochs=3,verbose=2)

	model.save('test_model.h5')


