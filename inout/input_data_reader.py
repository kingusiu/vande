import os
import h5py
import pandas as pd
import numpy as np
from config import *
from event_to_image_converter import *
from di_jet import *
from util.utility_fun import *


class InputDataReader():

    def __init__( self, filename='.' ):
        if not self.check_path(filename): return
        self.filename = filename
        self.constituents_key = 'jetConstituentsList'
        self.features_key = 'eventFeatures'


    def check_path(self, path_name ):
        if not ( os.path.isfile(path_name) or os.path.isdir( path_name ) ):
            print('no path with name ', path_name)
            return False
        return True

    ##
    # reads images from file, returns images of jet 1 and jet 2 as numpy arrays
    def read_images( self ):
        print('=== reading images from ', self.filename, ' ===')
        file_in = h5py.File( self.filename, 'r')
        data = file_in.get('images_j1_j2')
        data = np.asarray(data, dtype='float32')
        print('read ', data[0].shape[0], ' jet 1 images and ', data[1].shape[0], ' jet 2 images')
        return [data[0], data[1]]


    def read_events( self, key ):
        file_in = h5py.File(self.filename, 'r')
        print('reading key ', key, ' of file ', self.filename, ' with keys ', list(file_in.keys()))
        data = file_in.get(key)
        print('read data of shape ', data.shape)
        return data[()]


    def read_events_jet_constituents(self):
        data = self.read_events( self.constituents_key )
        return [data[:, 0, :, :], data[:, 1, :, :]]


    def read_events_jet_features(self):
        return np.asarray(self.read_events(self.features_key), dtype='float32')


    def read_events_convert_to_images(self):
        events_j1, events_j2 = self.read_events_jet_constituents()
        img_j1, img_j2 = convert_events_to_image_j1j2(events_j1, events_j2, config['image_size'])
        event_quantities = self.read_events_jet_features()
        di_jet = DiJet(event_quantities)
        # cut on mass
        img_j1, img_j2, event_quantities = filter_arrays_on_value([img_j1, img_j2, di_jet.data], di_jet.mass_jj(), config['mass_cut'])
        di_jet = DiJet(event_quantities)
        # normalize by pixel values from training
        img_j1, img_j2 = normalize_by_jet_pt(img_j1, img_j2, di_jet)
        return [img_j1, img_j2, di_jet]


    """
        read result file and return data as recarray
    """
    def read_events_results( self ):
        results = self.read_events( config['result_key'] )
        labels = h5py.File(self.filename, 'r').get('labels')
        labels = list(map( lambda x : x.decode(), labels))
        # convert data to recarray
        return np.core.records.fromarrays(results.transpose(), names=labels)


    def read_events_results_concatenate( self, result_dir ):
        if not self.check_path( result_dir ) : return
        file_list = os.listdir( result_dir )
        data = None
        for file in file_list:
            self.filename = file
            data_aux = self.read_events_results()
            data = np.append( data, data_aux, axis=0 ) if data else data_aux

        return data


class CaseInputDataReader( InputDataReader ):

    def read_images( self ):
        f = h5py.File( self.filename, 'r' )
        truth = np.asarray(f.get('truth_label')).flatten()
        jet1 = np.asarray(f.get('j1_images'))[ ..., np.newaxis ]
        jet2 = np.asarray(f.get('j2_images'))[ ..., np.newaxis ]
        #qcd_j1, qcd_j2 = filter_arrays_on_value([jet1,jet2],truth,0,operator.eq) # qcd label = 0
        #grav_j1, grav_j2 = filter_arrays_on_value([jet1,jet2],truth,1,operator.eq) # G -> ZZ label = 1
        qcd_j1, qcd_j2 = jet1[ truth == 0 ], jet2[ truth == 0 ]
        grav_j1, grav_j2 = jet1[ truth == 1 ], jet2[ truth == 1 ]
        return [qcd_j1,qcd_j2,grav_j1,grav_j2]


def read_dijet_features_to_dataframe( path ):
    data, labels = read_dijet_features( path )
    return pd.DataFrame( data, columns=labels )

def read_results_to_dataframe( path ):
    data, labels = read_results( path )
    return pd.DataFrame(data, columns=labels)