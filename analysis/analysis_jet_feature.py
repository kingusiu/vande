import skhep.math as hep
from functools import reduce
import numpy as np


# TODO: rename to 'scikit sample from dijet sample'
def dijet_sample_from_dijet_sample(sample):
    '''
    computing dijet with scikit hep package from dijet features given in input file (to obtain missing features like pt-jj)
    '''
    j1_sample = zip(sample['j1Pt'], sample['j1Eta'], [0.0] * len(sample), sample['j1M'])
    j2_sample = zip(sample['j2Pt'], sample['j1Eta'] - sample['DeltaEtaJJ'], sample['DeltaPhiJJ'], sample['j2M'])
    j1_sample = [jet_from_eta_phi_pt_m(event) for event in j1_sample]
    j2_sample = [jet_from_eta_phi_pt_m(event) for event in j2_sample]
    return [j1 + j2 for j1, j2 in zip(j1_sample, j2_sample)]


def jet_from_eta_phi_pt_m(data):
    pt, eta, phi, m = data if len(data) == 4 else np.append(data,0.0) # extract eta phi pt and m if given otherwise set m = 0
    jet = hep.vectors.LorentzVector()
    jet.setptetaphim(pt, eta, phi, m)
    return jet


def jet_sample_from_particle_sample(particle_sample):
    '''
    :param particles: N x 100 x 3 ( N .. number of events, 100 particles, 3 features (eta, phi, pt) )
    :return: N x 1 ( N events, each consisting of 1 jet )
    '''
    event_jets = []
    for event in particle_sample:
        particle_jets = [jet_from_eta_phi_pt_m(particle) for particle in event]
        event_jets.append(reduce(lambda x,y: x+y, particle_jets)) # sum all particle-jets to get event-jet

    return event_jets

