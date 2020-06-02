import os
from config import *
import setGPU

import inout.input_data_reader as idr
from vae.vae_prediction import *
import analysis_input.analysis_jet_image as aji

# ********************************************************
#       runtime params
# ********************************************************

run_n = 44

# ********************************************************
#       read in data ( jet constituents & jet features )
# ********************************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'qcd_side_new.h5' ))
train_img_j1, train_img_j2 = data_reader.read_images( )

img_analysis = aji.AnalysisJetImage('QCD train', do=['sampled_img', 'avg_img'], run=run_n)
img_analysis.analyze( [train_img_j1, train_img_j2] )

training_img = np.vstack([train_img_j1,train_img_j2])
np.random.shuffle( training_img )

# *******************************************************
#                       build and train
# *******************************************************

vae = VAE(run_n)
vae.build()

history = vae.fit( training_img, training_img, epochs=100, verbose=2 )
vae.plot_training( run=run_n )
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

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'G_to_WW_narrow_2p5TeV_new.h5' ) )
test_img_j1, test_img_j2 = data_reader.read_images( )

img_analysis = aji.AnalysisJetImage('G to WW na 2.5TeV', do=['sampled_img', 'avg_img'], run=run_n)
img_analysis.analyze( [test_img_j1, test_img_j2] )

# *******************************************************
#               predict signal
# *******************************************************

reco_img_j1 = vae.predict( test_img_j1 )
reco_img_j2 = vae.predict( test_img_j2 )    

# ********************************************
#               analyze
# ********************************************

img_analysis.update_name('G to WW na 2.5TeV reco')
img_analysis.analyze( [reco_img_j1, reco_img_j2] )
