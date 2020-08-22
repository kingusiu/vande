import tensorflow as tf
import numpy as np

import util.event_sample as es


root_dir = '/eos/home-k/kiwoznia/dev/autoencoder_for_anomaly/convolutional_VAE/data/events/qcd_sqrtshatTeV_13TeV_PU40_concat'

list_ds = tf.data.Dataset.list_files(root_dir+'/*')

for f in list_ds.take(20):
	print(f.numpy())

