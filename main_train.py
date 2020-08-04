import os
import setGPU
import numpy as np

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_model import VAE
from vae.vae_highres_model import VAE_HR
import inout.sample_factory as sf

# ********************************************************
#       runtime params
# ********************************************************

run_n = 49
data_sample = 'img-54'

experiment = ex.Experiment(run_n).setup(model_dir=True)
paths = sf.SamplePathFactory(experiment, data_sample)

# ********************************************************
#       read in training data ( images )
# ********************************************************

data_reader = idr.InputDataReader(paths.qcd_path)
train_img_j1, train_img_j2 = data_reader.read_images( )

# ********************************************************
#       prepare training data
# ********************************************************

training_img = np.vstack([train_img_j1,train_img_j2])
np.random.shuffle(training_img)

# *******************************************************
#                       build model
# *******************************************************

vae = VAE(run=run_n, model_dir=experiment.model_dir, input_size=54)
vae.build()

# *******************************************************
#                       train and save
# *******************************************************

vae.fit(training_img, training_img, epochs=100, verbose=2)
vae.save_model()
