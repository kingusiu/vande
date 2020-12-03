import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_hist( data, xlabel, ylabel, title, plotname='', legend=[], ylogscale=True ):
    fig = plt.figure( )
    plot_hist_on_axis( plt.gca(), data, xlabel, ylabel, title, legend, ylogscale )
    if legend:
        plt.legend()
    plt.tight_layout()
    fig.savefig('fig/' + plotname + '.png')
    plt.close()


def plot_hist_on_axis( ax, data, xlabel, ylabel='count', title='', legend=[], ylogscale=True ):
    bin_num = 70
    alpha = 0.85
    if ylogscale:
        ax.set_yscale('log', nonposy='clip')
    ax.hist( data, bins=bin_num, normed=True, alpha=alpha, histtype='stepfilled', label=legend )
    ax.set_ylabel( ylabel )
    ax.set_xlabel( xlabel )
    ax.set_title( title, fontsize=10 )
    ax.tick_params(axis='both', which='minor', labelsize=8)
    #ax.set_ylim(bottom=1e-7)


def plot_hist_2d( x, y, xlabel, ylabel, title, plotname=''):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im = plot_hist_2d_on_axis( ax, x, y, xlabel, ylabel, title )
    fig.colorbar(im[3])
    plt.tight_layout()
    fig.savefig('fig/' + plotname + '.png')
    #plt.close()
    plt.show()


def plot_hist_2d_on_axis( ax, x, y, xlabel, ylabel, title ):
    im = ax.hist2d(x, y, bins=100, norm=colors.LogNorm())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #ax.set_ylim(top=70.)
    return im


def plot_graph( data, xlabel, ylabel, title, plotname='', legend=[], ylogscale=True):
    fig = plt.figure()
    if ylogscale:
        plt.semilogy( data )
    else:
        plt.plot( data )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    if legend: plt.legend(legend, loc='upper right')
    plt.title( title )
    plt.tight_layout( )
    fig.savefig('fig/' + plotname + '_graph.png')
    plt.close()
