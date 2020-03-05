import numpy as np
import math
from util_plotting import *


class AnalysisEncoder( object ):

    def __init__( self, data_name ):
        self.data_name = data_name
        self.fig_name = data_name.replace(" ", "_")

        # ***********************************
        #          analyze latent space
        # ***********************************

    def analyze(self, jet1, jet2):
        z_mu_j1, z_log_sigma_j1 = jet1
        z_mu_j2, z_log_sigma_j2 = jet2
        self.plot_latent_space_1D_hist(z_mu_j1, z_log_sigma_j1, self.data_name + " j1", self.fig_name + "_j1")
        self.plot_latent_space_1D_hist(z_mu_j2, z_log_sigma_j2, self.data_name + " j2", self.fig_name + "_j2")
        self.plot_latent_space_2D_hist(z_mu_j1, z_log_sigma_j1, self.data_name + " j1", self.fig_name + "_j1")
        self.plot_latent_space_2D_hist(z_mu_j2, z_log_sigma_j2, self.data_name + " j2", self.fig_name + "_j2")


    def plot_latent_space_1D_hist(self, mu, log_sigma, title_suffix='data', file_suffix='data'):
        plot_hist(mu.flatten(), 'mu', 'fraction number events', 'mu distribution ' + title_suffix, 'latent_space_hist_1d_mu_' + file_suffix)
        plot_hist(log_sigma.flatten(), 'log sigma', 'fraction number events', 'log sigma distribution ' + title_suffix, 'latent_space_hist_1d_log_sigma_' + file_suffix)


    def plot_latent_space_2D_hist(self, mu, log_sigma, title_suffix="data", filename_suffix="data"):

        num_latent_dim = mu.shape[1]

        heatmaps = []
        xedges = []
        yedges = []

        # create 2d histogram for each latent space dimension
        for i in np.arange(num_latent_dim):
            heatmap, xedge, yedge = np.histogram2d(np.array(mu[:, i]), np.array(log_sigma[:, i]), bins=70)
            heatmaps.append(heatmap)
            xedges.append(xedge)
            yedges.append(yedge)

        heatmaps = np.asarray(heatmaps)
        xedges = np.asarray(xedges)
        yedges = np.asarray(yedges)

        min_hist_val = np.min(heatmaps[heatmaps > 0])  # find min value for log color bar clipping zero values
        max_hist_val = np.max(heatmaps)

        extent = [np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]

        nrows = int(round(math.sqrt(num_latent_dim)))
        ncols = math.ceil(math.sqrt(num_latent_dim))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))

        # plot all 2d histograms to one plot
        for d, ax in zip(np.arange(num_latent_dim), axs.flat[:num_latent_dim]):
            im = ax.imshow(heatmaps[d].T, extent=extent, origin='lower', norm=colors.LogNorm(vmin=min_hist_val, vmax=max_hist_val))
            ax.set_title('dim ' + str(d), fontsize='small')
            # fig.colorbar(im, ax=ax)

        for a in axs[:, 0]: a.set_ylabel('log sigma')
        for a in axs[-1, :]: a.set_xlabel('mu')
        for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
        if axs.size > num_latent_dim:
            for a in axs.flat[num_latent_dim:]: a.axis('off')

        plt.suptitle('mu vs sigma ' + title_suffix)
        # sns.jointplot(x=mu[:,0], y=log_sigma[:,0], kind='hex')
        cb = fig.colorbar(im)
        cb.set_label('count')
        plt.tight_layout()
        fig.savefig('fig/' + 'latent_space_hist_2d_' + filename_suffix + '.png')
        plt.close(fig)

