import tensorflow as tf
import util as ut
import numpy as np


class CustomLoss(tf.keras.losses.Loss):
	pass


x, y = ut.get_test_data_for_bin_classifier() 

model = ut.get_simple_dnn(input_shape=x.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x,y)
