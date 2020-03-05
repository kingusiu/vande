import numpy as np
import skhep.math as hep


class DiJet():

    def __init__( self, jet_quantities = [] ):
        self.data = jet_quantities
        self.data_names = np.array(["mJJ", "j1Pt", "j1Eta", "j1Phi", "j1M", "j1E", "j2Pt", "j2M", "j2E", "DeltaEtaJJ", "DeltaPhiJJ"], dtype='S')
        self.num_events = self.data.shape[0]

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

    def pt_jj(self):
        j1_feat, j2_features = zip( self.pt_j1(), self.eta_j1(), [0.0] * self.num_events, self.mass_j1() ), zip( self.pt_j2(), self.eta_j1()-self.deltaeta_jj(), self.deltaphi_jj(), self.mass_j2())
        return [ self.compute_pt_jj(j1,j2) for j1, j2 in zip(j1_feat,j2_features) ]
        # todo vectorize

    def compute_pt_jj(self, j1_feat, j2_feat):
        pt1, eta1, phi1, m1 = j1_feat
        pt2, eta2, phi2, m2 = j2_feat
        jet1 = hep.vectors.LorentzVector()
        jet2 = hep.vectors.LorentzVector()
        jet1.setptetaphim( pt1, eta1, phi1, m1 )
        jet2.setptetaphim( pt2, eta2, phi2, m2 )
        dijet = jet1 + jet2
        return dijet.pt

