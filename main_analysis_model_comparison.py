from collections import OrderedDict
import os

import analysis_main.main_analysis_losses as al
import analysis.analysis_roc as ar
import util.experiment as ex
import util.jet_sample as js
import config.sample_dict as sd
import inout.sample_factory as sf
import discriminator.loss_strategy as ls


# ********************************************************
#               runtime params
# ********************************************************

run_n_model1 = 46
run_n_model2 = 47

SM_sample = 'qcdSideReco'
BSM_samples = ['GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
strategies = ['s1', 's2', 's3', 's4', 's5', 'k1', 'k2']

all_sample_ids = [SM_sample] + BSM_samples
mass_centers = [1500, 1500, 2500, 2500, 3500, 3500, 4500, 4500]

# read JET IMAGE VAE model results
experiment = ex.Experiment(run_n_model1)
data_img_vae = sf.read_results_to_jet_sample_dict(all_sample_ids, experiment, mode='img-local')

# read 3D LOSS VAE model results
experiment = ex.Experiment(run_n_model2).setup(fig_dir=True)
data_3d_vae = sf.read_results_to_jet_sample_dict(all_sample_ids, experiment, mode='img-local')

for s in strategies:

    strategy = ls.loss_strategies[s]

    for BSM_sample, mass_center in zip(BSM_samples, mass_centers):
        _, binned_bg_img_vae, _  = ar.get_mjj_binned_sample(data_img_vae[SM_sample], mass_center)
        _, binned_sig_img_vae, _ = ar.get_mjj_binned_sample(data_img_vae[BSM_sample], mass_center)
        _, binned_bg_3D_vae, _ = ar.get_mjj_binned_sample(data_3d_vae[SM_sample], mass_center)
        _, binned_sig_3D_vae, _ = ar.get_mjj_binned_sample(data_3d_vae[BSM_sample], mass_center)

        neg_class_losses = [strategy(b) for b in [binned_bg_img_vae, binned_bg_3D_vae]]
        pos_class_losses = [strategy(s) for s in [binned_sig_img_vae, binned_sig_3D_vae]]
        legend = [binned_sig_img_vae.name + ' model ' + str(run_n_model1), binned_sig_img_vae.name + ' model ' + str(run_n_model2) ]

        ar.plot_roc(neg_class_losses, pos_class_losses, legend=legend, title='model comparison ROC binned strategy ' + strategy.title_str + ' ' + sd.sample_name[BSM_sample], log_x=True, plot_name='ROC_binned_logTPR_' + strategy.file_str + '_' + sd.file_names[BSM_sample], fig_dir='fig/run_'+str(run_n_model1)+'_vs_'+str(run_n_model2), xlim=10**(-3))





