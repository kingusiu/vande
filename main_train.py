import os
from config import *
#import setGPU
import numpy as np

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_model import VAE

# ********************************************************
#       runtime params
# ********************************************************

run_n = 0
experiment = ex.Experiment( run_n, model_dir=True )


# ********************************************************
#       read in training data ( images )
# ********************************************************

data_reader = idr.InputDataReader( os.path.join( config['input_dir'], 'background_small_img_ptnormal_bin32.h5' ))
train_img_j1, train_img_j2 = data_reader.read_images( )

# ********************************************************
#       prepare training data
# ********************************************************

training_img = np.vstack([train_img_j1,train_img_j2])
np.random.shuffle( training_img )

# *******************************************************
#                       build model
# *******************************************************

vae = VAE(run_n,experiment.model_dir)
vae.build()

# *******************************************************
#                       train and save
# *******************************************************

vae.fit( training_img, training_img, epochs=10, verbose=2 )
vae.save_model( run_n )
