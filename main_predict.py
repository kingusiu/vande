import os

from config import *
from vae.vae_model import VAE
import vae.losses as lo
import inout.input_data_reader as idr
import util.jet_sample as js
import util.experiment as ex


# ********************************************************
#               runtime params
# ********************************************************

run_n = 0
experiment = ex.Experiment( run_n, result_dir=True )

# ********************************************
#               read test data (images)
# ********************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_mjj_cut_concat_10K_pt_img.h5' ) )
test_sample = js.JetSample.from_feature_array('GtoWW3.5TevNa', *data_reader.read_dijet_features())
test_img_j1, test_img_j2 = data_reader.read_images( )

# ********************************************
#               load model
# ********************************************

vae = VAE()
vae.load( run_n )

# *******************************************************
#               predict test data
# *******************************************************

reco_img_j1, z_mean_j1, z_log_var_j1 = vae.predict_with_latent( test_img_j1 )
reco_img_j2, z_mean_j2, z_log_var_j2 = vae.predict_with_latent( test_img_j2 )

# *******************************************************
#               compute losses
# *******************************************************

losses_j1 = lo.compute_loss_of_prediction_mse_kl(test_img_j1, reco_img_j1, z_mean_j1, z_log_var_j1)
losses_j2 = lo.compute_loss_of_prediction_mse_kl(test_img_j2, reco_img_j2, z_mean_j2, z_log_var_j2)

# *******************************************************
#               add losses to DataSample and save
# *******************************************************

for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
    test_sample.add_feature(label,loss)
for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
    test_sample.add_feature(label,loss)

test_sample.dump( os.path.join(experiment.result_dir,test_sample.name+'_result.h5' ))

