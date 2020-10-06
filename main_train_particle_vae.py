import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import setGPU
import numpy as np
import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import vae.vae_particle as vap
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input as sdi
import sarewt.data_reader as dare
import pofah.util.converter as conv
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import pofah.util.utility_fun as utfu

# ********************************************************
#       runtime params
# ********************************************************

run_n = 102
beta = 0.01
loss = losses.make_threeD_kl_loss #losses.make_mse_kl_loss
reco_loss = losses.threeD_loss #losses.mse_loss
cartesian = False
experiment = expe.Experiment(run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       read in training data ( events )
# ********************************************************
sample_id = 'qcdSideAll'
data_reader = dare.DataReader(paths.sample_dir_path(sample_id))
# convert constituents from cylindrical to cartesian coordinates if needed
train_evts_j1j2 = conv.eppt_to_xyz(data_reader.read_constituents_from_dir()) if cartesian else data_reader.read_constituents_from_dir()

# ********************************************************
#       prepare training data
# ********************************************************

train_evts = np.vstack([train_evts_j1j2[:,0,:,:], train_evts_j1j2[:,1,:,:]])
np.random.shuffle(train_evts)
mean_stdev = utfu.get_mean_and_stdev(train_evts)

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=(100,3), z_sz=10, filter_ini_n=6, kernel_sz=3, loss=loss, reco_loss=reco_loss, batch_sz=128, beta=beta)
vae.build(mean_stdev)

# *******************************************************
#                       train and save
# *******************************************************

vae.fit(train_evts, epochs=300, verbose=2)
vae.plot_training(experiment.fig_dir)
vae.save(path=experiment.model_dir)