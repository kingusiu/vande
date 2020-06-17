import numpy as np
import pandas as pd
import os
import inout.input_data_reader as idr
import inout.result_writer as rw

class EventSample():

    def __init__(self, name, particles=None, jet_features=None, particle_feature_names=None):
        '''
        datastructure that holds set of N events with each having two components: data of particles and data of jet features
        :param name: name of the sample
        :param particles: particle features like eta, phi, pt (numpy array) (todo: extend to preprocessed form like images (implement subclass?))
        :param jet_features: N x F_n features (pandas dataframe)
        '''
        self.name = name
        self.file_name = self.name.replace(' ','_')+'.h5'
        self.particles = np.asarray(particles) # numpy array [ 2 (jets) x N events x 100 particles x 3 features ]
        self.particle_feature_names = particle_feature_names
        self.jet_features = pd.DataFrame(jet_features) # dataframe: names = columns

    @classmethod
    def from_input_file(cls,name,path):
        reader = idr.InputDataReader(path)
        particles, part_feature_names = reader.read_jet_constituents()
        jet_features = reader.read_dijet_features_to_df()
        return cls(name, np.stack(particles), jet_features, part_feature_names)

    def get_particles(self):
        return [self.particles[0],self.particles[1]]

    def dump(self,path):
        path = os.path.join(path,self.file_name)
        particles = np.stack((self.particles[0],self.particles[1]), axis=1) # particles in input files stored as ( N x 2 jets x 100 particles x 3 features )
        rw.write_event_sample_to_file(particles, self.jet_features.values, self.particle_feature_names, list(self.jet_features.columns), path)


