import sklearn.metrics as skl
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import numpy as np

import discriminator.loss_strategy as ls
import util.experiment as ex
import util.jet_sample as js
import config.sample_dict as sd

def get_label_and_score_arrays(neg_class_losses, pos_class_losses):
    labels = []
    losses = []

    for neg_loss, pos_loss in zip(neg_class_losses, pos_class_losses):
        labels.append(np.concatenate([np.zeros(len(neg_loss)), np.ones(len(pos_loss))]))
        losses.append(np.concatenate([neg_loss, pos_loss]))

    return [labels, losses]


def plot_roc(y_true_arr, loss_arr, legend=[], title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None):
    aucs = []
    fig = plt.figure(figsize=(5, 5))

    for y_true, loss, label in zip(y_true_arr, loss_arr, legend):
        fpr, tpr, threshold = skl.roc_curve(y_true, loss)
        aucs.append(skl.roc_auc_score(y_true, loss))
        plt.loglog(fpr, tpr, label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")")

    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.title(title)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight')

    return aucs


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

for BSM_sample in BSM_samples:

    neg_class_loss = [strategy( data[SM_sample] ) for strategy in ls.loss_strategies.values()]
    pos_class_losses = [strategy( data[BSM_sample] ) for strategy in ls.loss_strategies.values()]

    class_labels, losses = get_label_and_score_arrays( neg_class_loss, pos_class_losses ) # neg_class_loss array same for all pos_class_losses
    plot_roc( class_labels, losses, legend=legend, title='ROC '+sd.sample_name[BSM_sample], plot_name='ROC_'+sd.file_names[BSM_sample], fig_dir=experiment.fig_dir_event )