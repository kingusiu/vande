import os

from config import *
from vae.vae_model import VAE
import inout.input_data_reader as idr


# ********************************************************
#               runtime params
# ********************************************************

run_n = 44

# ********************************************
#               read test data (images)
# ********************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'G_to_WW_narrow_2p5TeV_new.h5' ) )
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
#               save test data results
# *******************************************************

