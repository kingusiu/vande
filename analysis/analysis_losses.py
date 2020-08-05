import analysis.analysis_bg_vs_sig as bgsig
import discriminator.loss_strategy as ls
import util.plotting as up
import pofah.sample_dict as sd


losses = ['j1TotalLoss','j2TotalLoss','j1RecoLoss','j2RecoLoss','j1KlLoss','j2KlLoss']
strategies = ['s1', 's2', 's3', 's4', 's5']

def analyze_losses(experiment, sample_dict, samples_to_analyze, plot_suffix=''):
    for loss in losses:
        bgsig.plot_feature(sample_dict, loss, samples_to_analyze, fig_dir=experiment.fig_dir_event, plot_suffix=plot_suffix)


def analyze_loss_strategies(experiment, sample_dict, samples_to_analyze, plot_suffix=''):
    legend = [ sample_dict[samp].name for samp in samples_to_analyze ]
    for strategy in ls.loss_strategies.values():
        combined_loss = [ strategy(sample_dict[samp]) for samp in samples_to_analyze]
        bgsig.plot_bg_vs_sig_distribution(combined_loss,xlabel=strategy.title_str,title=strategy.title_str+' distribution',fig_dir=experiment.fig_dir_event,plot_name='hist_'+strategy.file_str+'_'+plot_suffix,legend=legend)


def analyze_loss_strategies_keep_for_roc(experiment, sample_dict, samples_to_analyze, plot_suffix=''):
    legend = [ ls.loss_strategies[s].title_str for s in strategies]
    for sample in samples_to_analyze:
        title_suffix = sd.sample_name[sample]
        file_suffix = sd.file_names[sample]
        combined_losses = [strategy(sample_dict[sample]) for strategy in ls.loss_strategies.values()]
        up.plot_hist(combined_losses,xlabel='combined loss',legend=legend,title='combined loss strategies '+title_suffix,plot_name='hist_loss_strategies_'+plot_suffix+'_'+file_suffix,fig_dir=experiment.fig_dir_event)
