import h5py
import os
from config import *
from result import *

def write_results_array_to_file( results, labels, file_path ):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('results', data=results,  compression='gzip')
        f.create_dataset('eventFeatureNames', data=[l.encode("utf-8") for l in labels])
