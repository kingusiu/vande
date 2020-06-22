import os
from collections import OrderedDict

import analysis_data.analysis_losses as alo
import util.jet_sample as js
import util.experiment as ex
import config.sample_dict as sd


run_n = 45
experiment = ex.Experiment(run_n).setup(fig_dir=True)

SM_sample = 'qcdSideReco'
#BSM_samples = ['qcdSigReco', 'GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
BSM_samples = ['qcdSigReco', 'GtoWW15brReco', 'GtoWW25brReco', 'GtoWW35brReco', 'GtoWW45brReco']
#BSM_samples = ['qcdSigReco', 'GtoWW15naReco', 'GtoWW25naReco', 'GtoWW35naReco', 'GtoWW45naReco']

all_samples = [SM_sample] + BSM_samples

data = OrderedDict()
for sample_id in all_samples:
    data[sample_id] = js.JetSample.from_input_file(sample_id, os.path.join(experiment.result_dir, sd.file_names[sample_id]+'.h5'))

alo.analyze_losses( experiment, data, all_samples, 'broad_signal')