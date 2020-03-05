import os
from config import *
from util_plotting import *


def plot_loss_hist_total_reco_kl( background_loss, signal_loss, title, plotname, signal_label ):
    bg_total, bg_reco, bg_kl = background_loss
    sig_total, sig_reco, sig_kl = signal_loss
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    plot_hist_on_axis(ax1, [bg_total, sig_total], 'loss', 'fraction number events', 'total loss ' + title, ['QCD', signal_label])
    ax2 = fig.add_subplot(312, sharex=ax1)
    plot_hist_on_axis(ax2, [bg_reco, sig_reco], 'loss', 'fraction number events', 'reco loss ' + title, ['QCD', signal_label])
    ax3 = fig.add_subplot(313)
    plot_hist_on_axis(ax3, [bg_kl, sig_kl], 'loss', 'fraction number events', 'KL loss ' + title, ['QCD', signal_label])
    plt.legend()
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.join.path(config['plot_dir'], plotname + '.png'))
    plt.close()


def plot_loss_hist_all_strategies():
    pass


def plot_loss_vs_jet_feature( loss, jet_feature ):
    pass