import os
import setGPU

import pofah.util.experiment as ex
import pofah.util.event_sample as es
import vae.losses as lo
from vae.vae_3Dloss_model import VAE_3D
import pofah.sample_dict as sd
import config.config as co
import pofah.util.sample_factory as sf


# ********************************************************
#               runtime params
# ********************************************************

#test_samples = ['qcdSig', 'GtoWW15na', 'GtoWW15br', 'GtoWW25na', 'GtoWW25br', 'GtoWW45na', 'GtoWW45br']
#test_samples = ['qcdSide', 'GtoWW35na', 'GtoWW35br']
test_samples = ['qcdSig']

run_n = 101
data_sample = 'particle'

experiment = ex.Experiment(run_n).setup(result_dir=True)
paths = sf.SamplePathFactory(experiment, data_sample)


# ********************************************
#               load model
# ********************************************

vae = VAE_3D(run=run_n, model_dir=experiment.model_dir)
vae.load( )

for sample_id in test_samples:

    # ********************************************
    #               read test data (events)
    # ********************************************

    test_sample = es.EventSample.from_input_file(sample_id, paths.sample_path(sample_id))
    test_evts_j1, test_evts_j2 = test_sample.get_particles()

    # *******************************************************
    #               predict test data
    # *******************************************************

    print('predicting {}'.format(sd.sample_name[sample_id]))
    test_evts_j1_reco, z_mean_j1, z_log_var_j1 = vae.predict_with_latent(test_evts_j1)
    test_evts_j2_reco, z_mean_j2, z_log_var_j2 = vae.predict_with_latent(test_evts_j2)

    # *******************************************************
    #               compute losses
    # *******************************************************

    losses_j1 = lo.compute_loss_of_prediction_3D_kl(test_evts_j1, test_evts_j1_reco, z_mean_j1, z_log_var_j1)
    losses_j2 = lo.compute_loss_of_prediction_3D_kl(test_evts_j2, test_evts_j2_reco, z_mean_j2, z_log_var_j2)

    # *******************************************************
    #               add losses to DataSample and save
    # *******************************************************

    reco_sample = es.EventSample(sample_id + 'Reco', particles=[test_evts_j1_reco,test_evts_j2_reco], event_features=test_sample.get_event_features(), particle_feature_names=test_sample.particle_feature_names)

    for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
        reco_sample.add_event_feature(label, loss)
    for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
        reco_sample.add_event_feature(label, loss)

    # *******************************************************
    #               write predicted data
    # *******************************************************

    print('writing results for {} to {}'.format(sd.sample_name[sample_id],experiment.result_dir))
    reco_sample.dump(experiment.result_dir)
