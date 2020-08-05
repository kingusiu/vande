import os
from collections import OrderedDict

import analysis.analysis_losses as alo
import pofah.jet_sample as js
import pofah.util.experiment as ex
import pofah.sample_dict as sd
import pofah.util.sample_factory as sf

def analyze_losses( run_n, SM_sample_id, BSM_sample_ids, plot_suffix=''):

    experiment = ex.Experiment(run_n=run_n).setup(fig_dir=True)
    paths = sf.SamplePathFactory(experiment)  # 'default' datasample because reading only results

    all_samples = [SM_sample_id] + BSM_sample_ids

    data = OrderedDict()
    for sample_id in all_samples:
        data[sample_id] = js.JetSample.from_input_file(sample_id, paths.result_path(sample_id))

    alo.analyze_losses(experiment, data, all_samples, plot_suffix)

    alo.analyze_loss_strategies(experiment, data, all_samples, plot_suffix)