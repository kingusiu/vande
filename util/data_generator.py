import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu


class DataGenerator():

    def __init__(self, path, samples_in_parts_n=1e4, samples_max_n=None):
        self.path = path
        self.samples_in_parts_n = int(samples_in_parts_n) # samples_in_parts_n events from file parts
        self.samples_max_n = int(samples_max_n) if samples_max_n else None


    def constituents_to_input_samples(self, constituents): # -> np.ndarray
        samples = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
        np.random.shuffle(samples)
        return samples  


    def __call__(self): # -> generator object yielding (np.ndarray, np.ndarray)
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        print('[DataGenerator]: __call__()')
        
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        data_reader = dare.DataReader(self.path)

        samples_read_n = 0
        # loop through whole dataset, reading samples_in_parts_n events at a time
        for constituents in data_reader.generate_constituents_parts_from_dir(parts_n=self.samples_in_parts_n):
            samples = self.constituents_to_input_samples(constituents)
            indices = list(range(len(samples)))
            samples_read_n += len(samples)
            while indices:
                index = indices.pop(0)
                next_sample = samples[index] #.copy() 
                yield next_sample, next_sample  # x == y in autoencoder
            if self.samples_max_n is not None and (samples_read_n > self.samples_max_n):
                print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
                break


    def get_mean_and_stdev(self): # -> float, float
        '''
            get mean and standard deviation of input samples constituents (first 1 million events) 
        '''
        data_reader = dare.DataReader(self.path)

        constituents = data_reader.read_constituents_from_dir(max_n=1e6)
        constituents_j1j2 = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
        return utfu.get_mean_and_stdev(constituents_j1j2)
