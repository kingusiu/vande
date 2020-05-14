from input_data_reader import *
from vae_prediction import *
from analysis_encoder import *
from analysis_jet_image import *

# ********************************************************
#       read in data ( jet constituents & jet features )
# ********************************************************

data_reader = CaseInputDataReader( './data/BB_images_batch10_subset1000.h5' ) # AtoHZ_to_ZZZ_13TeV_PU40_concat.h5
qcd_img_j1, qcd_img_j2, grav_img_j1, grav_img_j2 = data_reader.read_images( )
AnalysisJetImage('BB images batch10 QCD').analyze( [qcd_img_j1, qcd_img_j2] )
AnalysisJetImage('BB images batch10 G to ZZ').analyze( [grav_img_j1, grav_img_j2] )


# *******************************************************
#                       predict
# *******************************************************

#dijet_latent, dijet_reco_img = predict( img_j1, img_j2, di_jet, 'results_run_0' )

# ********************************************
#               analyze
# ********************************************

#AnalysisEncoder('A to HZ to ZZZ').analyze( dijet_latent[0], dijet_latent[1])
#AnalysisJetImage('A to HZ to ZZZ reco').analyze( dijet_reco_img )
