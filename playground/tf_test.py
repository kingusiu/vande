import tensorflow as tf

@tf.function
def bar(vv):
	print(vv)
	return vv
	

def foo(v):
	for vv in v:
		print(vv.numpy())

v = tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices(v)
dataset.map(bar)
foo(dataset)
