import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu




def mask_training_cuts(constituents, features):
    ''' get mask for training cuts requiring a jet-pt > 200'''
    jetPt_cut = 200.
    idx_j1Pt, idx_j2Pt = 1, 6
    mask_j1 = features[:, idx_j1Pt] > jetPt_cut
    mask_j2 = features[:, idx_j2Pt] > jetPt_cut
    return mask_j1, mask_j2

def constituents_to_input_samples(constituents, mask_j1, mask_j2): # -> np.ndarray
        const_j1 = constituents[:,0,:,:][mask_j1]
        const_j2 = constituents[:,1,:,:][mask_j2]
        samples = np.vstack([const_j1, const_j2])
        np.random.shuffle(samples)
        return samples  

def events_to_input_samples(constituents, features):
    mask_j1, mask_j2 = mask_training_cuts(constituents, features)
    return constituents_to_input_samples(constituents, mask_j1, mask_j2)


class DataGenerator():

    def __init__(self, path, sample_part_n=1e4, sample_max_n=None, **cuts):
        ''' 
            sample_part_n ... number of events(!) read as chunk from file-data-generator (TODO: change to event_part_n)
            sample_max_n ... number of single jet samples as input into VAE (unpacked dijets)
        '''
        self.path = path
        self.sample_part_n = int(sample_part_n) # sample_part_n events from file parts
        self.sample_max_n = int(sample_max_n) if sample_max_n else None
        self.cuts = cuts


    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = dare.DataReader(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents, features in generator:
            samples = events_to_input_samples(constituents, features)
            indices = list(range(len(samples)))
            samples_read_n += len(samples)
            while indices:
                index = indices.pop(0)
                next_sample = samples[index] #.copy() 
                yield next_sample
            if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n):
                break
        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()


    def get_mean_and_stdev(self): # -> nd.array [num-features], nd.array [num-features]
        '''
            get mean and standard deviation of input samples constituents (first 1 million events) for each feature
        '''
        data_reader = dare.DataReader(self.path)

        constituents = data_reader.read_constituents_from_dir(read_n=int(1e6))
        constituents_j1j2 = np.vstack([constituents[:,0,:,:], constituents[:,1,:,:]])
        utfu.get_mean_and_stdev(constituents_j1j2)
        return utfu.get_mean_and_stdev(constituents_j1j2)


class CaseDataGenerator():

    def __init__(self, path, sample_part_n=1e4, sample_max_n=None, **cuts):
        ''' 
            sample_part_n ... number of events(!) read as chunk from file-data-generator (TODO: change to event_part_n)
            sample_max_n ... number of single jet samples as input into VAE (unpacked dijets)
        '''
        self.path = path
        self.sample_part_n = int(sample_part_n) # sample_part_n events from file parts
        self.sample_max_n = int(sample_max_n) if sample_max_n else None
        self.cuts = cuts


    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = dare.CaseDataReader(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents, features in generator:
            samples = events_to_input_samples(constituents[:,:,:,:3], features)
            indices = list(range(len(samples)))
            samples_read_n += len(samples)
            while indices:
                index = indices.pop(0)
                next_sample = samples[index] #.copy() 
                yield next_sample
            if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n):
                break
        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()


    def get_mean_and_stdev(self): # -> nd.array [num-features], nd.array [num-features]
        '''
            get mean and standard deviation of input samples constituents (first 1 million events) for each feature
        '''
        data_reader = dare.CaseDataReader(self.path)

        constituents = data_reader.read_constituents_from_dir(read_n=int(1e3))
        constituents_j1j2 = np.vstack([constituents[:,0,:,:3], constituents[:,1,:,:3]])

        return utfu.get_mean_and_stdev(constituents_j1j2)


class DataGeneratorMixedBgSig():

    def __init__(self, path_bg, path_sig, sample_bg_part_n, sample_sig_part_n, sample_bg_total_n=None, sample_sig_total_n=None):
        self.path_bg = path_bg
        self.path_sig = path_sig
        self.sample_bg_part_n = int(sample_bg_part_n)
        self.sample_sig_part_n = int(sample_sig_part_n)
        self.sample_bg_total_n = int(sample_bg_total_n) if sample_bg_total_n else None
        self.sample_sig_total_n = int(sample_sig_total_n) if sample_sig_total_n else None


    def __call__(self): # -> generator object yielding (np.ndarray, np.ndarray)

        # make background and signal sample generators
        generator_bg = dare.DataReader(self.path_bg).generate_constituents_parts_from_dir(parts_n=self.sample_bg_part_n) 
        generator_sig = dare.DataReader(self.path_sig).generate_constituents_parts_from_dir(parts_n=self.sample_sig_part_n)

        sig_every_n_bg_samples = sample_bg_total_n//sample_sig_total_n
 
        samples_read_bg_n, samples_read_sig_n = 0

        while True:
            bg_constituents = constituents_to_input_samples(next(generator_bg))
            sig_constituents = constituents_to_input_samples(next(generator_sig))
