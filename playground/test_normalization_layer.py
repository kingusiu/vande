import numpy as np
import tensorflow as tf

print(tf.__version__)

# input [N_examples X particles X features]
N_examples = 10000
num_part = 4
eta = np.random.rand(N_examples,num_part)*4
phi = np.random.rand(N_examples,num_part)*2
pt = np.random.rand(N_examples,num_part)*700
dat = np.stack((eta,phi,pt),axis=-1)

print('dat shape: ', dat.shape)
#print('-'*10 + 'dat'+'-'*10, dat)
input = tf.keras.Input(shape=(num_part,3))

# dataset normalization

norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(dat)
x = norm_layer(input)
output = tf.keras.layers.Lambda(lambda x: x * tf.sqrt(norm_layer.variance) + norm_layer.mean)(x)

model = tf.keras.Model(input,output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)
model.fit(dat,dat,batch_size=10,epochs=5,verbose=2)

# predict data
predicted = model.predict(dat)
print('original = mean: {}, var {}'.format(np.mean(dat,axis=(0,1)), np.var(dat,axis=(0,1)))) 
print('predicted = mean: {}, var {}'.format(np.mean(predicted,axis=(0,1)), np.var(predicted,axis=(0,1))))
np.testing.assert_allclose(dat,predicted)

print(norm_layer.weights)
print('*'*100)


# batch normalization

batch_norm_layer = tf.keras.layers.BatchNormalization(scale=False, center=False, epsilon=1e-7)
output = batch_norm_layer(input, training=True)

model = tf.keras.Model(input,output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)
model.fit(dat,dat,batch_size=2,epochs=5,verbose=2)


#print('-'*10 + 'out'+'-'*10, out)
print(batch_norm_layer.weights)


# predict data
predicted = model.predict(dat)
batch_norm_mean = batch_norm_layer.moving_mean.numpy()
batch_norm_var = batch_norm_layer.moving_variance.numpy()
print('batch norm mean: {}, var: {}'.format(batch_norm_mean, batch_norm_var))

# reconstruct data
predicted_reco = (predicted * np.sqrt(batch_norm_var)) + batch_norm_mean 
print('original = mean: {}, var {}'.format(np.mean(dat,axis=(0,1)), np.var(dat,axis=(0,1)))) 
print('predicted = mean: {}, var {}'.format(np.mean(predicted,axis=(0,1)), np.var(predicted,axis=(0,1)))) 
print('reconstr = mean: {}, var {}'.format(np.mean(predicted_reco,axis=(0,1)), np.var(predicted_reco,axis=(0,1)))) 
np.testing.assert_allclose(dat,predicted_reco)

# add unnormalizing as lambda layer

batch_norm_layer = tf.keras.layers.BatchNormalization(scale=False, center=False, epsilon=1e-7)
x = batch_norm_layer(input, training=True)
output = tf.keras.layers.Lambda(lambda x: x * tf.sqrt(batch_norm_layer.moving_variance) + batch_norm_layer.moving_mean)(x)

model = tf.keras.Model(input,output)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mean_squared_error)
model.fit(dat,dat,batch_size=10,epochs=5,verbose=2)

# predict data
predicted = model.predict(dat)
print('predicted = mean: {}, var {}'.format(np.mean(predicted,axis=(0,1)), np.var(predicted,axis=(0,1))))
print('original = mean: {}, var {}'.format(np.mean(dat,axis=(0,1)), np.var(dat,axis=(0,1)))) 
batch_norm_mean = batch_norm_layer.moving_mean.numpy()
batch_norm_var = batch_norm_layer.moving_variance.numpy()
print('batch norm mean: {}, var: {}'.format(batch_norm_mean, batch_norm_var))

np.testing.assert_allclose(dat,predicted)
