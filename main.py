import os
from config import *
#import setGPU

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_prediction import *
import analysis_input.analysis_jet_image as aji

# ********************************************************
#       runtime params
# ********************************************************

run_n = 1
experiment = ex.Experiment( run_n, model_dir=True, fig_dir=True )

# ********************************************************
#       read in data ( jet constituents & jet features )
# ********************************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'background_small_img_ptnormal_bin32.h5' ))
train_img_j1, train_img_j2 = data_reader.read_images( )

img_analysis = aji.AnalysisJetImage('QCD train', do=['sampled_img', 'avg_img'], fig_dir=experiment.fig_dir)
img_analysis.analyze( [train_img_j1, train_img_j2] )

training_img = np.vstack([train_img_j1,train_img_j2])
np.random.shuffle( training_img )

# *******************************************************
#                       build and train
# *******************************************************

vae = VAE(run_n)
vae.build()

history = vae.fit( training_img, training_img, epochs=100, verbose=2 )
vae.plot_training( experiment.fig_dir )
vae.save_model( run_n )

# *******************************************************
#                       predict
# *******************************************************

reco_img_j1 = vae.predict( train_img_j1 )
reco_img_j2 = vae.predict( train_img_j2 )

# ********************************************
#               analyze
# ********************************************

img_analysis.update_name('QCD train reco')
img_analysis.analyze( [reco_img_j1, reco_img_j2] )


# ********************************************
#               load signal
# ********************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_mjj_cut_concat_10K_pt_img.h5' ) )
test_img_j1, test_img_j2 = data_reader.read_images( )

img_analysis = aji.AnalysisJetImage('G to WW na 3.5TeV', do=['sampled_img', 'avg_img'], fig_dir=experiment.fig_dir)
img_analysis.analyze( [test_img_j1, test_img_j2] )

# *******************************************************
#               predict signal
# *******************************************************

reco_img_j1 = vae.predict( test_img_j1 )
reco_img_j2 = vae.predict( test_img_j2 )    

# ********************************************
#               analyze
# ********************************************

img_analysis.update_name('G to WW na 3.5TeV reco')
img_analysis.analyze( [reco_img_j1, reco_img_j2] )
