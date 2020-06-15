import numpy as np
import os
import pathlib
from util_plotting import *
from config import *


class AnalysisJetImage( object ):

    def __init__(self, data_name, do=['all_px_hist', 'single_px_hist', 'sampled_img', 'avg_img'], fig_dir=None ):

        self.analyses_dict = {
            'all_px_hist': self.plot_pixel_histogram,
            'single_px_hist': self.plot_sampled_single_pixel_hist,
            'sampled_img': self.plot_sampled_images,
            'avg_img': self.plot_average_image
        }

        self.data_name = data_name
        self.fig_name = data_name.replace(" ","_")
        self.analyses = do #[ self.analyses_dict[analysis] for analysis in do]
        self.fig_dir = fig_dir


    def update_name(self, new_name):
        self.data_name = new_name
        self.fig_name = new_name.replace(" ", "_")

    # ***********************************
    #          analyze jet images
    # ***********************************

    def analyze(self, pixel_dijet):

        pixel_j1, pixel_j2 = pixel_dijet

        if 'all_px_hist' in self.analyses:
            self.plot_pixel_histogram( pixel_j1, pixel_j2 )

        if 'single_px_hist' in self.analyses:
            self.plot_sampled_single_pixel_hist( pixel_j1, 'j1')
            self.plot_sampled_single_pixel_hist( pixel_j2, 'j2')

        if 'sampled_img' in self.analyses:
            self.plot_sampled_images( pixel_j1, 'j1')
            self.plot_sampled_images( pixel_j2, 'j2')

        if 'avg_img' in self.analyses:
            self.plot_average_image( pixel_j1, 'j1')
            self.plot_average_image( pixel_j2, 'j2')


    # *****       pixel plots     *****

    def plot_pixel_histogram(self, pixel_j1, pixel_j2):

        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        plot_hist_on_axis(ax1, pixel_j1.flatten(), 'pixel value', 'fraction number events','pixel value histogram j1 ' + self.data_name)

        ax2 = fig.add_subplot(212, sharex=ax1)
        plot_hist_on_axis(ax2, pixel_j2.flatten(), 'pixel value', 'fraction number events','pixel value histogram j2 ' + self.data_name)

        plt.tight_layout(rect=(0, 0.05, 1, 1))
        if self.fig_dir:
            fig.savefig(os.path.join(self.fig_dir, 'pixel_hist_1d_j1_j2_' + self.fig_name + '.png'))
        plt.close()


    def plot_sampled_single_pixel_hist(self, image_stack, jet_num ):

        num_samples_per_dim = 5

        i_idx = [2, 9, 16, 23, 29]  # np.sort(np.random.randint(0,CONFIG['numBins'],num_samples_per_dim))
        j_idx = i_idx  # np.sort(np.random.randint(0,CONFIG['numBins'],num_samples_per_dim))

        cart_prod = [(a, b) for a in i_idx for b in j_idx]  # create index grid

        fig, axs = plt.subplots(nrows=num_samples_per_dim, ncols=num_samples_per_dim, sharex=True, sharey=True, figsize=(9, 9))

        bins = np.linspace(0.0, np.amax(image_stack), 40)

        for (i, j), ax in zip(cart_prod, axs.flat):
            ax.hist(image_stack[:, i, j], bins=bins)
            ax.set_yscale('log', nonposy='clip')
            ax.set_title('pixel (' + str(i) + ',' + str(j) + ')', fontsize='small')

        plt.ylim(bottom=1.0)
        for a in axs[:, 0]: a.set_ylabel('number events')
        for a in axs[-1, :]: a.set_xlabel('pixel value')
        plt.suptitle('sampled single pixel dist ' + jet_num + ' ' + self.data_name)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.draw()
        if self.fig_dir:
            fig.savefig( os.path.join( self.fig_dir, 'pixel_single_sampled_hist_' + jet_num + '_' + self.fig_name + '.png' ) )
        plt.close(fig)


    # *****       image plots     *****

    def plot_sampled_images(self, image_stack, jet_num ):

        num_samples_per_dim = 4

        if not hasattr( self, 'sampled_img_idx' ):
            self.sampled_img_idx = np.random.randint(len(image_stack), size=num_samples_per_dim * num_samples_per_dim)

        fig, axs = plt.subplots(nrows=num_samples_per_dim, ncols=num_samples_per_dim, figsize=(9, 9))

        vmax = np.max(image_stack)
        vmin = np.min(image_stack)

        for i, ax in zip(self.sampled_img_idx, axs.flat):
            # ax.pcolormesh(image_stack[i,:,:], norm=colors.LogNorm(vmin=logLowBound))
            im = ax.pcolormesh(image_stack[i, :, :, :].reshape(image_stack.shape[1], image_stack.shape[2]), norm=colors.SymLogNorm(linthresh=1e-5, vmin=vmin, vmax=vmax))  # drop last dimension of 32x32x1
            ax.set_title('image ' + str(i), fontsize='small')
            fig.colorbar(im, ax=ax)

        plt.suptitle('sampled images ' + jet_num + ' ' + self.data_name)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        if self.fig_dir:
            fig.savefig(os.path.join( self.fig_dir, 'image_sampled_' + jet_num + '_' + self.fig_name + '.png' ) )
        plt.close(fig)


    def plot_average_image( self, images, jet_num ):

        images_sum = images.sum(axis=0)
        images_sum = np.divide(images_sum, images.shape[0])
        images_sum = images_sum.reshape(images.shape[1], images.shape[2])
        vmin = np.min(images_sum)
        vmax = np.max(images_sum)
        plt.title('average image ' + jet_num + ' ' + self.data_name)
        plt.pcolormesh(images_sum, norm=colors.SymLogNorm(linthresh=1e-5, vmin=vmin, vmax=vmax))
        plt.colorbar()
        if self.fig_dir:
            plt.savefig(os.path.join( self.fig_dir, 'images_average_' + jet_num + '_' + self.fig_name + '.png'))
        plt.close()

