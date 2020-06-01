import h5py
import os
from config import *
from result import *


def get_complete_result_array( jet_quantities, losses_j1, losses_j2 ):

    losses_j1, losses_j2 = np.transpose(losses_j1), np.transpose(losses_j2)
    return np.hstack((jet_quantities, np.hstack((losses_j1, losses_j2))))


def write_results_for_analysis_to_file(jet_quantities, losses_j1, losses_j2, filename):

    full_result_set = get_complete_result_array(jet_quantities, losses_j1, losses_j2)

    print('writing data to', os.path.join(config['result_dir'], filename ))
    file_out = h5py.File(os.path.join(config['result_dir'], filename), "w")
    file_out.create_dataset(config['result_key'], data=full_result_set, compression='gzip')
    colnames = Result([]).data_names
    file_out.create_dataset('labels', data=colnames)
    file_out.close()

