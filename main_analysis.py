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
#               pick analysis
# ********************************************************

do_loss_analysis = False
do_binned_roc_analysis = True

# ********************************************************
#               runtime params
# ********************************************************

run_n = 0

SM_sample = 'qcdSideReco'
BSM_samples = ['GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
strategies = ['s1', 's2', 's3', 's4', 's5']

all_sample_ids = [SM_sample] + BSM_samples


if do_loss_analysis:
    al.analyze_losses(run_n,SM_sample, BSM_samples)

if do_binned_roc_analysis:

    mass_centers = [1500, 1500, 2500, 2500, 3500, 3500, 4500, 4500]

    # read JET IMAGE VAE model results
    run_n = 46
    experiment = ex.Experiment(run_n)
    data_img_vae = sf.read_results_to_jet_sample_dict(all_sample_ids, experiment)

    # read 3D LOSS VAE model results
    run_n = 45
    experiment = ex.Experiment(run_n).setup(fig_dir=True)
    data_3d_vae = sf.read_results_to_jet_sample_dict(all_sample_ids, experiment)

    for s in strategies:

        strategy = ls.loss_strategies[s]

        for BSM_sample, mass_center in zip(BSM_samples, mass_centers):
            _, binned_bg_img_vae, _  = ar.get_mjj_binned_sample(data_img_vae[SM_sample], mass_center)
            _, binned_sig_img_vae, _ = ar.get_mjj_binned_sample(data_img_vae[BSM_sample], mass_center)
            _, binned_bg_3D_vae, _ = ar.get_mjj_binned_sample(data_img_vae[SM_sample], mass_center)
            _, binned_sig_3D_vae, _ = ar.get_mjj_binned_sample(data_img_vae[BSM_sample], mass_center)

            neg_class_losses = [strategy(b) for b in binned_bg]
            pos_class_losses = [strategy(s) for s in binned_sig]
            legend = [s.name for s in binned_sig]

            ar.plot_roc(neg_class_losses, pos_class_losses, legend=legend, title='ROC binned strategy ' + strategy.title_str + ' ' + sd.sample_name[BSM_sample], plot_name='ROC_binned_' + strategy.file_str + '_' + sd.file_names[BSM_sample], fig_dir=experiment.fig_dir_event)





