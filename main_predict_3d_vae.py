import os

import util.experiment as ex
import util.event_sample as es
from vae.vae_3Dloss_model import VAE_3D
from config import *

# ********************************************************
#               runtime params
# ********************************************************

run_n = 5
experiment = ex.Experiment(run_n).setup(result_dir=True)

# ********************************************
#               read test data (events)
# ********************************************

test_sample = es.EventSample.from_input_file('RS Graviton WW br 3.0TeV',os.path.join(config['input_dir'],'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_concat_10K.h5'))
test_evts_j1, test_evts_j2 = test_sample.get_particles()

# ********************************************
#               load model
# ********************************************

vae = VAE_3D()
vae.load( experiment.model_dir )

# *******************************************************
#               predict test data
# *******************************************************

test_evts_j1_reco = vae.predict(test_evts_j1)
test_evts_j2_reco = vae.predict(test_evts_j2)

reco_sample = es.EventSample(test_sample.name + ' reco', particles=[test_evts_j1_reco,test_evts_j2_reco], particle_feature_names=test_sample.particle_feature_names)
reco_sample.dump(experiment.result_dir)
