import os
import matplotlib.pyplot as plt

import config.sample_dict as sd
import util.jet_sample as js
import util.experiment as ex
import analysis_data.analysis_bg_vs_sig as bgsig

run_n = 45
experiment = ex.Experiment(run_n).setup(fig_dir=True)

SM_sample = 'qcdSideReco'
#BSM_samples = ['qcdSigReco', 'GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
#BSM_samples = ['qcdSigReco', 'GtoWW15brReco', 'GtoWW25brReco', 'GtoWW35brReco', 'GtoWW45brReco']
BSM_samples = ['qcdSigReco', 'GtoWW15naReco', 'GtoWW25naReco', 'GtoWW35naReco', 'GtoWW45naReco']

all_samples = [SM_sample] + BSM_samples

data = {}
for sample_id in all_samples:
    data[sample_id] = js.JetSample.from_input_file(sample_id, os.path.join(experiment.result_dir, sd.file_names[sample_id]+'.h5'))

losses = ['j1TotalLoss','j2TotalLoss','j1RecoLoss','j2RecoLoss','j1KlLoss','j2KlLoss']

for loss in losses:
    bgsig.plot_feature(data, loss, all_samples, fig_dir=experiment.fig_dir_event,plot_suffix='narrow_signals')
