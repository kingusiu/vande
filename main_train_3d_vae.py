import os
import setGPU
import numpy as np

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_3Dloss_model import VAE_3D
import config.sample_dict as sd
import config.config as co
import inout.sample_factory as sf


# ********************************************************
#       runtime params
# ********************************************************

run_n = 101
data_sample = 'particle'

experiment = ex.Experiment(run_n).setup(model_dir=True, fig_dir=True)
paths = sf.SamplePathFactory(experiment, data_sample)


# ********************************************************
#       read in training data ( events )
# ********************************************************

data_reader = idr.InputDataReader(paths.qcd_path)
train_evts_j1, train_evts_j2 = data_reader.read_jet_constituents(with_names=False)

# ********************************************************
#       prepare training data
# ********************************************************

training_evts = np.vstack([train_evts_j1, train_evts_j2])
np.random.shuffle(training_evts)

# *******************************************************
#                       build model
# *******************************************************

vae = VAE_3D(run=run_n,model_dir=experiment.model_dir)
vae.build()

# *******************************************************
#                       train and save
# *******************************************************

history = vae.fit(training_evts, training_evts, epochs=3, verbose=2)
vae.plot_training(experiment.fig_dir)
vae.save_model()