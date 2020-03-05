from input_data_reader import *
from config import *
from vae_model import *
from result_writer import *


def predict( img_j1, img_j2, di_jet, result_file_name, run = 0 ):

    # ********************************************
    #               load model
    # ********************************************

    vae = VAE()
    vae.load( run )

    # ********************************************
    #               predict
    # ********************************************

    z_mean_j1, z_log_var_j1, z_j1 = vae.encoder.predict(img_j1, batch_size=vae.batch_size)
    img_reco_j1 = vae.decoder.predict(z_j1, batch_size=vae.batch_size)

    z_mean_j2, z_log_var_j2, z_j2 = vae.encoder.predict(img_j2, batch_size=vae.batch_size)
    img_reco_j2 = vae.decoder.predict(z_j2, batch_size=vae.batch_size)


    # compute losses
    loss_j1 = compute_loss_of_prediction( img_j1, img_reco_j1, z_mean_j1, z_log_var_j1 )
    loss_j2 = compute_loss_of_prediction( img_j2, img_reco_j2, z_mean_j2, z_log_var_j2 )

    # write results
    write_results_for_analysis_to_file(di_jet.data, loss_j1, loss_j2, result_file_name)

    # TODO: put return in some nicer structure
    # return [ dijet-latent-space, dijet-img-reconstruction]
    return [[[z_mean_j1, z_log_var_j1], [z_mean_j2, z_log_var_j2]], [img_reco_j1,img_reco_j2]]
