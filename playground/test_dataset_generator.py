import tensorflow as tf
import numpy as np


def generator():

	for i in range(10):
		n = np.random.randint(5,15)
		yield np.random.random(size=3)


ds = tf.data.Dataset.from_generator(generator, tf.float32)

for x in ds:
	print(x)
	print(x.shape)