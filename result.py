import numpy as np

class Result():

    def __init__( self, result_array = [] ):
        self.data = result_array
        self.data_names = np.array(["mJJ", "j1Pt", "j1Eta", "j1Phi", "j1M", "j1E", "j2Pt", "j2M", "j2E", "DeltaEtaJJ", "DeltaPhiJJ", "j1TotalLoss", "j1RecoLoss", "j1KlLoss", "j2TotalLoss", "j2RecoLoss", "j2KlLoss"], dtype='S')

    def mass_jj(self):
        return self.data[:,0]

    def pt_j1(self):
        return self.data[:,1]

    def eta_j1(self):
        return self.data[:,2]

    def phi_j1(self):
        return self.data[:,3]

    def mass_j1(self):
        return self.data[:,4]

    def e_j1(self):
        return self.data[:,5]

    def pt_j2(self):
        return self.data[:,6]

    def mass_j2(self):
        return self.data[:,7]

    def e_j2(self):
        return self.data[:,8]

    def deltaeta_jj(self):
        return self.data[:,9]

    def deltaphi_jj(self):
        return self.data[:, 10]

    def loss_total_j1(self):
        return self.data[:,11]

    def loss_reco_j1(self):
        return self.data[:,12]

    def loss_kl_j1(self):
        return self.data[:,13]

    def loss_total_j2(self):
        return self.data[:,14]

    def loss_reco_j2(self):
        return self.data[:,15]

    def loss_kl_j2(self):
        return self.data[:,16]



