import os
from config import *
#import setGPU
import numpy as np

import util.experiment as ex
import inout.input_data_reader as idr
import analysis.analysis_constituents as ac
from vae.vae_3Dloss_model import VAE_3D

# ********************************************************
#       runtime params
# ********************************************************

run_n = 4
experiment = ex.Experiment(run_n).setup(model_dir=True, fig_dir=True)


# ********************************************************
#       read in training data ( events )
# ********************************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'background_small.h5' ))
train_evts_j1, train_evts_j2 = data_reader.read_jet_constituents( )

particle_analysis = ac.AnalysisConstituents('QCD train', fig_dir=experiment.fig_dir_event)
particle_analysis.analyze([train_evts_j1, train_evts_j2])

# ********************************************************
#       prepare training data
# ********************************************************

training_evts = np.vstack([train_evts_j1, train_evts_j2])
np.random.shuffle( training_evts )

# *******************************************************
#                       build model
# *******************************************************

vae = VAE_3D(run=run_n,model_dir=experiment.model_dir)
vae.build()

# *******************************************************
#                       train and save
# *******************************************************

vae.fit( training_evts, training_evts, epochs=50, verbose=2 )
vae.save_model( run_n )

# *******************************************************
#                   predict training set
# *******************************************************

reco_train_j1 = vae.predict(train_evts_j1)
reco_train_j2 = vae.predict(train_evts_j2)

particle_analysis.update_name('QCD train reco')
particle_analysis.analyze([reco_train_j1,reco_train_j2])

# ********************************************
#               load signal
# ********************************************

data_reader = idr.InputDataReader(os.path.join( config['input_dir'],'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV_concat_10K.h5'))
test_evts_j1, test_evts_j2 = data_reader.read_jet_constituents( )

particle_analysis = ac.AnalysisConstituents('GRS narr 3.0TeV', fig_dir=experiment.fig_dir_event)
particle_analysis.analyze( [test_evts_j1, test_evts_j2] )


# *******************************************************
#               predict signal
# *******************************************************

reco_test_evts_j1 = vae.predict( test_evts_j1 )
reco_test_evts_j2 = vae.predict( test_evts_j2 )

particle_analysis.update_name('GRS narr 3.0TeV reco')
particle_analysis.analyze([reco_test_evts_j1,reco_test_evts_j2])

print('-- finished --')