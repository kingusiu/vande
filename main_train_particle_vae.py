import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import vae.vae_particle as vap
import pofah.path_constants.sample_dict_file_parts_input as sdi
import sarewt.data_reader as dare


# ********************************************************
#       runtime params
# ********************************************************

run_n = 102

experiment = ex.Experiment(run_n).setup(model_dir=True, fig_dir=True)
paths = sf.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       read in training data ( events )
# ********************************************************
sample_id = 'qcdSideAll'
data_reader = dare.DataReader(paths.sample_dir_path(sample_id))
train_evts_j1, train_evts_j2 = data_reader.read_constituents_from_dir()

# ********************************************************
#       prepare training data
# ********************************************************

training_evts = np.vstack([train_evts_j1, train_evts_j2])
np.random.shuffle(training_evts)
mean, std_dev = ut.get_mean_and_std(training_evts)

# *******************************************************
#                       build model
# *******************************************************

vae = vap.VAEparticle(input_shape=(100,3), z_sz=10, filter_n=6, kernel_sz=3, loss=losses.make_mse_kl_loss, batch_sz=128, beta=0.01)
vae.build(mean, std_dev)

# *******************************************************
#                       train and save
# *******************************************************

vae.fit(training_evts, training_evts, epochs=100, verbose=2)
vae.plot_training(experiment.fig_dir)
vae.save_model(path=experiment.model_dir)