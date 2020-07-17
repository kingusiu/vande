import os
import h5py
import pandas as pd
import config.config as co
from inout.event_to_image_converter import *
import util.utility_fun as ut

default_jet_feature_names = ['mJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j1E', 'j2Pt', 'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']

class InputDataReader():

    def __init__( self, path='.' ):
        if not self.check_path(path): return
        self.path = path
        self.particles_key = 'jetConstituentsList'
        self.particles_names_key = 'particleFeatureNames'
        self.jet_features_key = 'eventFeatures'
        self.jet_feature_names_key = 'eventFeatureNames'


    def check_path(self, path_name ):
        if not ( os.path.isfile(path_name) or os.path.isdir( path_name ) ):
            print('no path with name ', path_name)
            return False
        return True

    ##
    # reads images from file, returns images of jet 1 and jet 2 as numpy arrays
    def read_images( self ):
        print('=== reading images from ', self.path, ' ===')
        with h5py.File( self.path, 'r') as file_in:
            data = np.asarray(file_in.get('images_j1_j2'), dtype='float32')
            print('read ', data[0].shape[0], ' jet 1 images and ', data[1].shape[0], ' jet 2 images')
            return [data[0], data[1]]


    def read_data_multikey(self, *keys):
        data = []
        with h5py.File(self.path, 'r') as f:
            if not keys:
                keys = f.keys()
            for k in keys:
                data.append(np.asarray(f.get(k)))
        return data

    def read_data(self,key):
        with h5py.File(self.path,'r') as f:
            return np.asarray(f.get(key))

    def read_string_data(self,key):
        with h5py.File(self.path,'r') as f:
            return [ n.decode("utf-8") for n in np.asarray(f.get(key)) ]

    def read_jet_constituents(self, with_names=True):
        data = np.asarray(self.read_data( self.particles_key ), dtype='float32')
        if with_names:
            return [data[:, 0, :, :], data[:, 1, :, :]], self.read_string_data(self.particles_names_key)
        return [data[:, 0, :, :], data[:, 1, :, :]]

    def read_dijet_feature_names(self):
        feature_names = self.read_data(self.jet_feature_names_key)
        return default_jet_feature_names if feature_names.size == 0 else [ n.decode("utf-8") for n in feature_names] # some files don't have feature names saved

    def read_dijet_features(self, with_names=True):
        if with_names:
            return [np.asarray(self.read_data(self.jet_features_key), dtype='float32'), self.read_dijet_feature_names()]
        return np.asarray(self.read_data(self.jet_features_key), dtype='float32')

    def read_dijet_features_to_df(self):
        features, names = self.read_dijet_features()
        return pd.DataFrame(features,columns=names)


    def read_events_convert_to_images(self):
        events_j1, events_j2 = self.read_jet_constituents()
        dijet_features, dijet_feature_names = self.read_dijet_features()
        # cut on mass
        mjj_idx = dijet_feature_names.index('mJJ')
        events_j1, events_j2, dijet_features = ut.filter_arrays_on_value( events_j1, events_j2, dijet_features, filter_arr=dijet_features[:,mjj_idx], filter_val=co.config['mass_cut'] )

        img_j1, img_j2 = convert_events_to_image(events_j1, events_j2, co.config['image_size'])
        # normalize by pixel values from training
        img_j1, img_j2 = normalize_by_jet_pt(img_j1, img_j2, dijet_features, dijet_feature_names )
        return [img_j1, img_j2, dijet_features, dijet_feature_names]


    def read_events_results_concatenate( self, result_dir ):
        if not self.check_path( result_dir ) : return
        file_list = os.listdir( result_dir )
        data = None
        for file in file_list:
            self.filename = file
            data_aux = self.read_results()
            data = np.append( data, data_aux, axis=0 ) if data else data_aux

        return data


class CaseInputDataReader( InputDataReader ):

    def read_images( self ):
        f = h5py.File( self.path, 'r' )
        truth = np.asarray(f.get('truth_label')).flatten()
        jet1 = np.asarray(f.get('j1_images'))[ ..., np.newaxis ]
        jet2 = np.asarray(f.get('j2_images'))[ ..., np.newaxis ]
        #qcd_j1, qcd_j2 = filter_arrays_on_value([jet1,jet2],truth,0,operator.eq) # qcd label = 0
        #grav_j1, grav_j2 = filter_arrays_on_value([jet1,jet2],truth,1,operator.eq) # G -> ZZ label = 1
        qcd_j1, qcd_j2 = jet1[ truth == 0 ], jet2[ truth == 0 ]
        grav_j1, grav_j2 = jet1[ truth == 1 ], jet2[ truth == 1 ]
        return [qcd_j1,qcd_j2,grav_j1,grav_j2]


def read_dijet_features_to_dataframe( path ):
    reader = InputDataReader(path)
    data, labels = reader.read_dijet_features_with_names( path )
    return pd.DataFrame( data, columns=labels )

def read_results_to_dataframe( path ):
    reader = InputDataReader(path)
    data, labels = reader.read_results( )
    return pd.DataFrame(data, columns=labels)