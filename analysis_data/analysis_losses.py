import os
import matplotlib.pyplot as plt

import config.sample_dict as sd
import util.jet_sample as js
import util.experiment as ex


def plot_bg_vs_sig_distribution(data, bins=100, xlabel='x', ylabel='counts', title='bg vs sig distribution', legend=[], normed=True, ylogscale=True, fig_dir=None, plot_name='bg_vs_sig_dist', legend_loc='best'):
    fig = plt.figure(figsize=(6, 4))
    alpha = 0.5
    histtype = 'stepfilled'
    if ylogscale:
        plt.yscale('log')

    for i, dat in enumerate(data):
        if i > 0:
            histtype = 'step'
            alpha = 1.0
        plt.hist(dat, bins=bins, normed=normed, alpha=alpha, histtype=histtype, label=legend[i])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'))


run_n = 45
experiment = ex.Experiment(run_n).setup(fig_dir=True)

SM_sample = 'qcdSideReco'
BSM_samples = ['GtoWW35naReco', 'GtoWW35brReco']

all_samples = [SM_sample] + BSM_samples

data = {}
for sample_id in all_samples:
    data[sample_id] = js.JetSample.from_input_file(sample_id, os.path.join(experiment.result_dir, sd.file_names[sample_id]+'.h5'))

labels = [sd.sample_name[id] for id in all_samples]
total_losses_j1 = [sample['j1TotalLoss'] for sample in data.values()]
total_losses_j2 = [sample['j2TotalLoss'] for sample in data.values()]
kl_losses_j1 = [sample['j1KlLoss'] for sample in data.values()]
kl_losses_j2 = [sample['j2KlLoss'] for sample in data.values()]


plot_bg_vs_sig_distribution(total_losses_j1,legend=labels,fig_dir=experiment.fig_dir_event,plot_name='bg_vs_sig_hist_L1_total.png',title='total loss j1 distribution BG vs SIG',xlabel='total loss jet1')
plot_bg_vs_sig_distribution(total_losses_j2,legend=labels,fig_dir=experiment.fig_dir_event,plot_name='bg_vs_sig_hist_L2_total.png',title='total loss j2 distribution BG vs SIG',xlabel='total loss jet2')
plot_bg_vs_sig_distribution(kl_losses_j1,legend=labels,fig_dir=experiment.fig_dir_event,plot_name='bg_vs_sig_hist_L1_kl.png',title='KL loss j1 distribution BG vs SIG',xlabel='kl loss jet1')
plot_bg_vs_sig_distribution(kl_losses_j2,legend=labels,fig_dir=experiment.fig_dir_event,plot_name='bg_vs_sig_hist_L2_kl.png',title='KL loss j2 distribution BG vs SIG',xlabel='kl loss jet2')

