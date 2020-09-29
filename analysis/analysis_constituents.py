import matplotlib.pyplot as plt
import os

from analysis.analysis import Analysis
import util_plotting as up

class AnalysisConstituents(Analysis):

    def __init__(self, data_name, do=['eta','phi','pt'], fig_dir='fig'):
        super(AnalysisConstituents, self).__init__(data_name,do,fig_dir)
        self.d_eta_idx, self.d_phi_idx, self.pt_idx = range(3)

    def analyze(self,data):

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))

        for jet in (0,1):
            for idx, name in enumerate(self.do):
                dat = data[jet][:,:,idx]
                up.plot_hist_on_axis(axs[idx,jet],dat.flatten(),xlabel=name+' jet'+str(jet+1))

        plt.suptitle(r'distribution particles '+ ' '.join(self.do) + ' ' + self.data_name)
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if self.fig_dir:
            fig.savefig(os.path.join( self.fig_dir, 'hist_'+'_'.join(self.do) + self.fig_name + '.png' ) )
        plt.close(fig)
