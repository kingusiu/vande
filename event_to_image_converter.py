import numpy as np
from config import *

"""
    find bin borders for delta-eta range & delta-phi range
"""
def bin_eta_phi( bin_n ):
    minAngle = -0.8;
    maxAngle = 0.8
    return np.linspace(minAngle, maxAngle, num=bin_n)


"""
    convert single jet to image
"""
def bin_eta_phi_pt_to_image( events, imageShape, bin_borders ):
    print('==== converting events to images ====')
    inputBlockShape = (events.shape[0], imageShape[0], imageShape[1], 1)
    images = np.zeros(inputBlockShape, dtype='float32')

    for eventNo, event in enumerate(events):  # for each event (100x3) populate eta-phi binned image with pt values
        # bin eta and phi of event event
        binIdxEta = np.digitize(event[:, 0], bin_borders, right=True) - 1  # binning starts with 0
        binIdxPhi = np.digitize(event[:, 1], bin_borders, right=True) - 1
        for particle in range(event.shape[0]):
            images[eventNo, binIdxEta[particle], binIdxPhi[particle]] += event[particle, 2]  # add pt to bin of jet image

    return images


"""
    convert j1 and j2 to images each
"""
def convert_events_to_image( events_j1, events_j2, bin_n ):
    bin_borders = bin_eta_phi(bin_n)
    return [bin_eta_phi_pt_to_image(events_j1, (bin_n, bin_n), bin_borders),bin_eta_phi_pt_to_image(events_j2, (bin_n, bin_n), bin_borders)]


"""
    normalize jet images by maximum pixel value in background side dataset
"""
def normalize_by_max_pixel( images_j1, images_j2 ):
    images_j1 = images_j1 / (config['max_pixel'] - config['min_pixel'])
    images_j2 = images_j2 / (config['max_pixel'] - config['min_pixel'])
    return [ images_j1, images_j2 ]


"""
    normalize jet images by pt of jet
"""


def normalize_by_jet_pt( images_j1, images_j2, jet_features, labels ):
    idx_ptj1 = labels.index('j1Pt')
    idx_ptj2 = labels.index('j2Pt')
    images_j1 = np.divide(images_j1, jet_features[:, idx_ptj1, None, None])
    images_j2 = np.divide(images_j2, jet_features[:, idx_ptj2, None, None])
    return [images_j1, images_j2]
