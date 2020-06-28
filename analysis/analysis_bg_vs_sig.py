import matplotlib.pyplot as plt
import os


def plot_feature( sample_dict, feature_name, sample_names=None, fig_dir=None, plot_suffix=None ):
    if not sample_names:
        sample_names = sample_dict.keys()
    legend = [ sample_dict[s].name for s in sample_names ]
    feature = [ sample_dict[s][feature_name] for s in sample_names ]
    plot_bg_vs_sig_distribution(feature,legend=legend,xlabel=feature_name,title=r'distribution '+feature_name, fig_dir=fig_dir, plot_name='hist_'+feature_name+'_'+plot_suffix)


def plot_bg_vs_sig_distribution(data, bins=100, xlabel='x', ylabel='frac', title='bg vs sig distribution', legend=[], normed=True, ylogscale=True, fig_dir=None, plot_name='bg_vs_sig_dist', legend_loc='best', first_is_bg=True):
    '''
    plots feature distribution treating first data-array as backround and rest of arrays as signal
    :param data: list/array of N elements where first element is assumed to be background and elements 2..N-1 assumed to be signal. all elements = array of length M
    '''
    fig = plt.figure(figsize=(6, 4))
    alpha = 0.4
    histtype = 'stepfilled'
    if ylogscale:
        plt.yscale('log')

    switch_style = 0 if first_is_bg is True else -1

    for i, dat in enumerate(data):
        if i > switch_style:
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
    plt.close(fig)
