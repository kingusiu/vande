import setGPU

from vae.vae_model import VAE
from vae.vae_highres_model import VAE_HR
import vae.losses as lo
import pofah.util.input_data_reader as idr
import pofah.util.sample_factory as sf
import pofah.jet_sample as js
import pofah.util.experiment as ex


# ********************************************************
#               runtime params
# ********************************************************

run_n = 4
data_sample = 'img-local-54'

experiment = ex.Experiment(run_n).setup(result_dir=True)
paths = sf.SamplePathFactory(experiment,data_sample)

# ********************************************
#               load model
# ********************************************

vae = VAE(run=run_n, model_dir=experiment.model_dir)
vae.load()

# ********************************************
#               read test data (images)
# ********************************************

#sample_ids = ['qcdSide', 'qcdSig', 'GtoWW15na', 'GtoWW15br', 'GtoWW25na', 'GtoWW25br', 'GtoWW35na', 'GtoWW35br', 'GtoWW45na', 'GtoWW45br']
sample_ids = ['GtoWW25br', 'GtoWW35na']

for sample_id in sample_ids:

    data_reader = idr.InputDataReader(paths.sample_path(sample_id))
    test_img_j1, test_img_j2 = data_reader.read_images( )


    # *******************************************************
    #               predict test data
    # *******************************************************

    print('-'*10, 'predicting', '-'*10)
    reco_img_j1, z_mean_j1, z_log_var_j1 = vae.predict_with_latent( test_img_j1 )
    reco_img_j2, z_mean_j2, z_log_var_j2 = vae.predict_with_latent( test_img_j2 )

    # *******************************************************
    #               compute losses
    # *******************************************************

    print('-'*10, 'computing losses', '-'*10)
    losses_j1 = lo.compute_loss_of_prediction_mse_kl(test_img_j1, reco_img_j1, z_mean_j1, z_log_var_j1, input_size=54)
    losses_j2 = lo.compute_loss_of_prediction_mse_kl(test_img_j2, reco_img_j2, z_mean_j2, z_log_var_j2, input_size=54)

    # *******************************************************
    #               add losses to DataSample and save
    # *******************************************************

    predicted_sample = js.JetSample.from_feature_array(sample_id, *data_reader.read_dijet_features())
    
    for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
        predicted_sample.add_feature(label,loss)
    for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
        predicted_sample.add_feature(label,loss)

    predicted_sample.dump(paths.result_path(sample_id + 'Reco'))

