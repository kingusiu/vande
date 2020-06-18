import os
import setGPU
import numpy as np

import util.experiment as ex
import inout.input_data_reader as idr
from vae.vae_3Dloss_model import VAE_3D
import config.sample_dict as sd

# ********************************************************
#       runtime params
# ********************************************************

train_sample = 'qcdSide'
input_path = os.path.join(sd.base_dir_events,sd.file_names[train_sample]+'_mjj_cut_concat_1.2M.h5') # os.path.join( config['input_dir'], 'background_small.h5' )

run_n = 55
experiment = ex.Experiment( run_n ).setup(model_dir=True)


# ********************************************************
#       read in training data ( events )
# ********************************************************

data_reader = idr.InputDataReader(input_path)
train_evts_j1, train_evts_j2 = data_reader.read_jet_constituents(with_names=False)

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

history = vae.fit( training_evts, training_evts, epochs=100, verbose=2 )
vae.plot_training( experiment.fig_dir )
vae.save_model( run_n )
