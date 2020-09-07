import os
import setGPU
import tensorflow as tf

import pofah.util.experiment as ex
import pofah.util.event_sample as es
import vae.losses as lo
from vae.vae_3Dloss_model import VAE_3D
import config.config as co
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_input as sdi 
import pofah.path_constants.sample_dict_file_parts_reco as sdr 


# ********************************************************
#               runtime params
# ********************************************************

#test_samples = ['GtoWW15na', 'GtoWW15br', 'GtoWW25na', 'GtoWW25br', 'GtoWW35na', 'GtoWW35br', 'GtoWW45na', 'GtoWW45br']
#test_samples = ['GtoWW35na', 'GtoWW35br']
#test_samples = ['qcdSig']
test_samples = ['qcdSigBis']

run_n = 101

experiment = ex.Experiment(run_n=run_n)

# ********************************************
#               load model
# ********************************************

vae = VAE_3D(run=run_n, model_dir=experiment.model_dir)
vae.load( )

input_paths = sf.SamplePathDirFactory(sdi.path_dict)
result_paths = sf.SamplePathDirFactory(sdr.path_dict).extend_base_path(experiment.run_dir)

for sample_id in test_samples:

    # ********************************************
    #               read test data (events)
    # ********************************************


    list_ds = tf.data.Dataset.list_files(input_paths.sample_dir_path(sample_id)+'/*')

    for file_path in list_ds:

        file_name = file_path.numpy().decode('utf-8').split(os.sep)[-1]
        test_sample = es.EventSample.from_input_file(sample_id, file_path.numpy().decode('utf-8'))
        test_evts_j1, test_evts_j2 = test_sample.get_particles()
        print('{}: {} j1 evts, {} j2 evts'.format(file_path.numpy().decode('utf-8'), len(test_evts_j1), len(test_evts_j2)))


        # *******************************************************
        #               predict test data
        # *******************************************************

        print('predicting {}'.format(sdi.path_dict['sample_name'][sample_id]))
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

        print('writing results for {} to {}'.format(sdr.path_dict['sample_name'][reco_sample.name], os.path.join(result_paths.sample_dir_path(reco_sample.name), file_name)))
        reco_sample.dump(os.path.join(result_paths.sample_dir_path(reco_sample.name), file_name))
