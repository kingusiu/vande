import matplotlib.pyplot as plt
import os

def plot_hist(data, bins=100, xlabel='x', ylabel='num frac', title='histogram', plot_name='', fig_dir=None, legend=[],ylogscale=True, normed=True, ylim=None, legend_loc='best', xlim=None):
    fig = plt.figure(figsize=(6, 4))
    plot_hist_on_axis(plt.gca(), data, bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend, ylogscale=ylogscale, normed=normed, ylim=ylim, xlim=xlim)
    if legend:
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'))
    plt.close()


def plot_hist_on_axis(ax, data, bins, xlabel, ylabel, title, legend=[], ylogscale=True, normed=True, ylim=None, xlim=None):
    if ylogscale:
        ax.set_yscale('log', nonposy='clip')
    counts, edges, _ = ax.hist(data, bins=bins, normed=normed, histtype='step', label=legend)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        print('setting xlim to ', xlim)
        ax.set_xlim(xlim)


