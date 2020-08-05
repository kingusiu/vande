from collections import OrderedDict
import os

import analysis.analysis_roc as ar
import discriminator.loss_strategy as ls
import pofah.sample_dict as sd
import pofah.jet_sample as js
import pofah.util.experiment as ex


strategies = ['s1', 's2', 's3', 's4', 's5']

legend = [ls.loss_strategies[s].title_str for s in strategies]

run_n = 45
experiment = ex.Experiment(run_n).setup(fig_dir=True)

SM_sample = 'qcdSideReco'
BSM_samples = ['qcdSigReco', 'GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']

all_samples = [SM_sample] + BSM_samples


data = OrderedDict()
for sample_id in all_samples:
    data[sample_id] = js.JetSample.from_input_file(sample_id, os.path.join(experiment.result_dir, sd.file_names[sample_id]+'.h5'))

# plot standard ROC for all strategies
for BSM_sample in BSM_samples:

    neg_class_losses = [strategy( data[SM_sample] ) for strategy in ls.loss_strategies.values()]
    pos_class_losses = [strategy( data[BSM_sample] ) for strategy in ls.loss_strategies.values()]

    ar.plot_roc( neg_class_losses, pos_class_losses, legend=legend, title='ROC '+sd.sample_name[BSM_sample], plot_name='ROC_'+sd.file_names[BSM_sample], fig_dir=experiment.fig_dir_event )

# plot binned ROC
BSM_samples = ['GtoWW15brReco','GtoWW15naReco','GtoWW25brReco','GtoWW25naReco','GtoWW35brReco','GtoWW35naReco','GtoWW45brReco','GtoWW45naReco']
mass_centers = [1500,1500,2500,2500,3500,3500,4500,4500]
strategies = ['s3','s4','s5']

for s in strategies:

    strategy = ls.loss_strategies[s]

    for BSM_sample, mass_center in zip(BSM_samples,mass_centers):

        binned_bg = ar.get_mjj_binned_sample(data[SM_sample], mass_center)
        binned_sig = ar.get_mjj_binned_sample(data[BSM_sample], mass_center)

        neg_class_losses = [ strategy( b ) for b in binned_bg ]
        pos_class_losses = [ strategy( s ) for s in binned_sig ]
        legend = [ s.name for s in binned_sig ]

        ar.plot_roc(neg_class_losses, pos_class_losses, legend=legend, title='ROC binned strategy '+ strategy.title_str+' '+ sd.sample_name[BSM_sample], plot_name='ROC_binned_'+strategy.file_str+'_'+ sd.file_names[BSM_sample], fig_dir=experiment.fig_dir_event)

