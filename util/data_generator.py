import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu


class DataGenerator():

	def __init__(self, path, batch_sz=1024):
		self.data_reader = dare.DataReader(path)
		self.batch_sz = batch_sz

	def get_next_sample_chunk(self, constituents):
		samples_in_file = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
		np.random.shuffle(samples_in_file)
		return samples_in_file	

	def __call__(self):
		'''
			generate data samples of size batch_sz
		'''
		
		samples = []

		# loop through whole dataset
		for constituents in self.data_reader.read_constituents_parts_from_dir(min_n=self.batch_sz):
			samples.extend(self.get_next_sample_chunk(constituents))
			while len(samples) >= self.batch_sz:
				samples_batch, samples = np.asarray(samples[:self.batch_sz]), samples[self.batch_sz:]
				yield (samples_batch, samples_batch) # x == y in autoencoder
		# last batch: if events left in samples, pad with start to batch_sz
		if samples:
			generator = self.data_reader.read_constituents_parts_from_dir(min_n=self.batch_sz)
			samples.extend(self.get_next_sample_chunk(next(generator)))
			samples_batch = np.asarray(samples[:self.batch_sz])
			yield (samples_batch, samples_batch)


	def get_mean_and_stdev(self):
		'''
			get mean and standard deviation of input samples constituents (first 1 million events) 
		'''
		constituents = self.data_reader.read_constituents_from_dir(max_n=1e6)
		constituents_j1j2 = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
		return utfu.get_mean_and_stdev(constituents_j1j2)
