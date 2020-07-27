import tensorflow as tf
import numpy as np
from sklearn import preprocessing

def get_mean_and_std(dat):
	std = np.nanstd(dat,axis=(0,1))
	std[std == 0.0] = 1.0 # handle zeros
	return np.nanmean(dat,axis=(0,1)), std


class Normalize(tf.keras.layers.Layer):
	pass

class Model():

	def run(self, dat, mean, std):
		input = tf.keras.Input(shape=dat.shape[1:])
		normalized = tf.keras.layers.Lambda(lambda xx: (xx-mean)/std)(input)
		unnormalized = tf.keras.layers.Lambda(lambda xx: (xx*std)+mean)(normalized)

		model = tf.keras.Model(input,unnormalized)
		model.compile(optimizer='adam',loss=tf.keras.losses.mse)

		model.fit(dat,dat,batch_size=10,epochs=5,verbose=2)

		model.save('test_model.h5')

	def load(self, dat):
		loaded_model = tf.keras.models.load_model('test_model.h5')
		print('-'*10,'loaded model, predicting...')
		return loaded_model.predict(dat)


# input [N_examples X particles X features]
N_examples = 5
num_part = 4
eta = np.random.rand(N_examples,num_part)*4
phi = np.random.rand(N_examples,num_part)*2
pt = np.random.rand(N_examples,num_part)*700
dat = np.stack((eta,phi,pt),axis=-1)

print('dat shape: ', dat.shape)


mean, std = get_mean_and_std(dat)

model = Model()
#model.run(dat, mean, std)
predicted = model.load(dat)
assert(np.allclose(dat, predicted))

exit()



dat_mean, dat_std = get_mean_and_std(dat)
print('np mean: {}, np std {}'.format(dat_mean, dat_std))
dat_normalized = (dat - dat_mean) / dat_std
mean = np.nanmean(dat_normalized,axis=(0,1))
std = np.nanstd(dat_normalized,axis=(0,1))
print('norm mean: ', mean)
assert(np.allclose(mean, 0))
print('norm std: ', std)
assert(np.allclose(std, 1.0))
print(dat_normalized)
dat_unnormalized = (dat_normalized * dat_std) + dat_mean
assert(np.allclose(dat,dat_unnormalized))
print('-'*10, 'dat original', '-'*10)
print(dat)
print('-'*10, 'dat unnorm', '-'*10)
print(dat_unnormalized)


exit()


