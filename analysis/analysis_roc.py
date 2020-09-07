import sklearn.metrics as skl
import matplotlib.pyplot as plt
import os
import numpy as np

import pofah.jet_sample as js


def get_label_and_score_arrays(neg_class_losses, pos_class_losses):
    labels = []
    losses = []

    for neg_loss, pos_loss in zip(neg_class_losses, pos_class_losses):
        labels.append(np.concatenate([np.zeros(len(neg_loss)), np.ones(len(pos_loss))]))
        losses.append(np.concatenate([neg_loss, pos_loss]))

    return [labels, losses]


def plot_roc(neg_class_losses, pos_class_losses, legend=[], title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None, xlim=None, log_x=True):

    class_labels, losses = get_label_and_score_arrays(neg_class_losses, pos_class_losses) # neg_class_loss array same for all pos_class_losses

    aucs = []
    fig = plt.figure(figsize=(5, 5))

    for y_true, loss, label in zip(class_labels, losses, legend):
        fpr, tpr, threshold = skl.roc_curve(y_true, loss)
        aucs.append(skl.roc_auc_score(y_true, loss))
        if log_x:
            plt.loglog(tpr, 1./fpr, label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")")
        else:
            plt.semilogy(tpr, 1./fpr, label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")")

    plt.grid()
    if xlim:
        plt.xlim(left=xlim)
    plt.xlabel('True positive rate')
    plt.ylabel('1 / False positive rate')
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.title(title)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight')
    plt.close(fig)

    return aucs


def get_mjj_binned_sample(sample, mjj_peak, window_pct=20):
    left_edge, right_edge = mjj_peak * (1. - window_pct / 100.), mjj_peak * (1. + window_pct / 100.)

    left_bin = sample[sample['mJJ'] < left_edge]
    center_bin = sample[(sample['mJJ'] >= left_edge) & (sample['mJJ'] <= right_edge)]
    right_bin = sample[sample['mJJ'] > right_edge]

    left_bin_ds = js.JetSample(sample.title() + ' mJJ < ' + str(left_edge / 1000), left_bin)
    center_bin_ds = js.JetSample(sample.title() + ' ' + str(left_edge / 1000) + ' <= mJJ <= ' + str(right_edge / 1000),
                                  center_bin)
    right_bin_ds = js.JetSample(sample.title() + ' mJJ > ' + str(right_edge / 1000), right_bin)

    return [left_bin_ds, center_bin_ds, right_bin_ds]

def plot_binned_roc( sample_dict, ):
    pass


