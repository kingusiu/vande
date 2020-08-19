import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

for e in dataset:
	print(e.numpy())

root_dir = '/eos/home-k/kiwoznia/dev/autoencoder_for_anomaly/convolutional_VAE/data/events/qcd_sqrtshatTeV_13TeV_PU40'

