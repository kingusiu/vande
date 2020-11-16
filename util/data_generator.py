import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu


class DataGenerator():

	def __init__(self, path, batch_sz=1024):
		self.data_reader = dare.DataReader(path)
		self.batch_sz = batch_sz

	def __call__():
		'''
			generate data samples of size batch_sz
		'''
		samples = []
		for constituents in self.data_reader.read_constituents_parts_from_dir(max_sz_mb=1000):
			samples_file = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
			samples.extend(samples_file)
			while len(samples) >= batch_sz:
				samples_batch, samples = np.asarray(samples[:batch_sz]), samples[batch_sz:]
				np.random.shuffle(samples_batch)
				yield samples_batch

	def get_mean_and_stdev(self):
		'''
			get mean and standard deviation of input samples constituents (first 1 million events) 
		'''
		constituents = self.data_reader.read_constituents_from_dir(max_n=1e6)
		constituents_j1j2 = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
		return utfu.get_mean_and_stdev(constituents_j1j2)
