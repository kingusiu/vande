import os

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_3Dloss_model import VAE_3D
from config import *

# ********************************************************
#               runtime params
# ********************************************************

run_n = 5
experiment = ex.Experiment( run_n, result_dir=True )

# ********************************************
#               read test data (events)
# ********************************************

data_reader = idr.InputDataReader(os.path.join( config['input_dir'],'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_concat_10K.h5'))
test_evts_j1, test_evts_j2 = data_reader.read_jet_constituents( )

# ********************************************
#               load model
# ********************************************

vae = VAE_3D()
vae.load( run_n )

# *******************************************************
#               predict test data
# *******************************************************

test_evts_j1_reco = vae.predict(test_evts_j1)
test_evts_j2_reco = vae.predict(test_evts_j2)