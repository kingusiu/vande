import numpy as np
import pandas as pd
import os
import pofah.util.input_data_reader as idr
import pofah.util.result_writer as rw
import pofah.sample_dict as sd


class EventSample():

    def __init__(self, name, particles=None, event_features=None, particle_feature_names=None):
        '''
        datastructure that holds set of N events with each having two components: data of particles and data of jet features
        :param name: name of the sample
        :param particles: particle features like eta, phi, pt (numpy array) (todo: extend to preprocessed form like images (implement subclass?))
        :param event_features: N x F_n features (pandas dataframe)
        '''
        self.name = name
        self.file_name = sd.file_names[self.name]+'.h5'
        self.particles = np.asarray(particles) # numpy array [ 2 (jets) x N events x 100 particles x 3 features ]
        self.particle_feature_names = particle_feature_names
        self.event_features = pd.DataFrame(event_features) # dataframe: names = columns

    @classmethod
    def from_input_file(cls,name,path):
        reader = idr.InputDataReader(path)
        particles, part_feature_names = reader.read_jet_constituents()
        jet_features = reader.read_dijet_features_to_df()
        return cls(name, np.stack(particles), jet_features, part_feature_names)

    def get_particles(self):
        return [self.particles[0],self.particles[1]]

    def get_event_features(self):
        return self.event_features

    def add_event_feature(self, label, value):
        self.event_features[label] = value

    def dump(self,path):
        path = os.path.join(path,self.file_name)
        particles = np.stack((self.particles[0],self.particles[1]), axis=1) # particles in input files stored as ( N x 2 jets x 100 particles x 3 features )
        rw.write_event_sample_to_file(particles, self.event_features.values, self.particle_feature_names, list(self.event_features.columns), path)


