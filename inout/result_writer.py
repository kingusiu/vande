import h5py
import os
import config.config as co
from result import *

def write_results_array_to_file( results, labels, file_path ):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('results', data=results,  compression='gzip')
        f.create_dataset('eventFeatureNames', data=[l.encode("utf-8") for l in labels])

def write_event_sample_to_file( particles, event_features, particle_feature_names, event_feature_names, path ):
    '''
    write particles, particle_feature_names, jet_features, jet_feature_names to file
    '''
    particles_key = 'jetConstituentsList'
    particles_names_key = 'particleFeatureNames'
    jet_features_key = 'eventFeatures'
    jet_feature_names_key = 'eventFeatureNames'

    with h5py.File(path,'w') as f:
        f.create_dataset(particles_key,data=particles,compression='gzip',dtype='float32')
        f.create_dataset(particles_names_key, data=[n.encode('utf-8') for n in particle_feature_names])
        f.create_dataset(jet_features_key, data=event_features, compression='gzip', dtype='float32')
        f.create_dataset(jet_feature_names_key, data=[n.encode('utf-8') for n in event_feature_names])

