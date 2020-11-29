import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu


class DataGenerator():

	def __init__(self, path, read_n=10e3):
		self.data_reader = dare.DataReader(path)
		self.read_n = int(read_n) # read_n events from file parts

	def constituents_to_input_samples(self, constituents):
		samples = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
		np.random.shuffle(samples)
		return samples	

	def __call__(self):
		'''
			generate single(!) data-sample (batching done in tf.Dataset)
		'''
		# loop through whole dataset, reading read_n events at a time
		for constituents in self.data_reader.generate_constituents_parts_from_dir(min_n=self.read_n):
			samples = self.constituents_to_input_samples(constituents)
			indices = list(range(len(samples)))
			while indices:
				index = indices.pop(0)
                yield samples[index], samples[index]  # x == y in autoencoder


	def get_mean_and_stdev(self):
		'''
			get mean and standard deviation of input samples constituents (first 1 million events) 
		'''
		constituents = self.data_reader.read_constituents_from_dir(max_n=1e6)
		constituents_j1j2 = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
		return utfu.get_mean_and_stdev(constituents_j1j2)
