import numpy as np
import tensorflow as tf


def get_test_data_for_vae(*args):
	if not args: args = (100,7,3) # 100 samples, 7 elements, with 3 features each
	return np.random.random(size=args), np.random.randint(5,size=args[0])

def get_test_data_for_bin_classifier(n_samples=100, n_features=7):
	x = np.random.random(size=(n_samples, n_features))
	y = np.random.randint(2, size=n_samples)
	return x, y

def get_simple_dnn(input_shape=(20,)):
	inputs = tf.keras.Input(shape=input_shape)
	x = tf.keras.layers.Dense(128, activation='relu')(inputs)
	x = tf.keras.layers.Dense(64, activation='relu')(x)
	x = tf.keras.layers.Dense(32, activation='relu')(x)
	outputs = tf.keras.layers.Dense(1, activation='relu')(x)
	model = tf.keras.Model(inputs,outputs)
	return model

def get_simple_autoencoder(input_shape=(100,3)):
	inputs = tf.keras.Input(shape=input_shape)
	x = tf.keras.layers.Flatten()(inputs)
	x = tf.keras.layers.Dense(50, activation='relu')(x)
	x = tf.keras.layers.Dense(10, activation='relu')(x)
	x = tf.keras.layers.Dense(3, activation='relu')(x)
	x = tf.keras.layers.Dense(10, activation='relu')(x)
	x = tf.keras.layers.Dense(50, activation='relu')(x)
	x = tf.keras.layers.Dense(np.prod(input_shape), activation='relu')(x)
	outputs = tf.keras.layers.Reshape(input_shape)(x)
	model = tf.keras.Model(inputs, outputs)
	# model.compile(optimizer='adam',loss='mse', experimental_run_tf_function=False) 
	model.compile(optimizer='adam',loss='mse') 
	return model
