import analysis_data.analysis_bg_vs_sig as bgsig

losses = ['j1TotalLoss','j2TotalLoss','j1RecoLoss','j2RecoLoss','j1KlLoss','j2KlLoss']

def analyze_losses( experiment, sample_dict, samples_to_analyze, plot_suffix='' ):
    for loss in losses:
        bgsig.plot_feature(sample_dict, loss, samples_to_analyze, fig_dir=experiment.fig_dir_event, plot_suffix=plot_suffix)


