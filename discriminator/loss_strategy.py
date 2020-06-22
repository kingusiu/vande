import numpy as np
from collections import OrderedDict

def combine_loss_l1( x ):
    """ L1 > LT """
    return x['j1TotalLoss']

def combine_loss_l2( x ):
    """ L2 > LT """
    return x['j2TotalLoss']

def combine_loss_sum( x ):
    """ L1 + L2 > LT """
    return x['j1TotalLoss'] + x['j2TotalLoss']

def combine_loss_max( x ):
    """ L1 | L2 > LT """
    return np.maximum(x['j1TotalLoss'],x['j2TotalLoss'])

def combine_loss_min( x ):
    """ L1 & L2 > LT """
    return np.minimum(x['j1TotalLoss'],x['j2TotalLoss'])


class LossStrategy():

    def __init__(self, loss_fun, title_str, file_str):
        self.fun = loss_fun
        self.title_str = title_str
        self.file_str = file_str

    def __call__(self, x):
        return self.fun(x)


loss_strategies = OrderedDict({ 's1' : LossStrategy(combine_loss_l1, 'L1 > LT', 'l1_loss'),
                     's2': LossStrategy(combine_loss_l2, 'L2 > LT', 'l2_loss'),
                     's3': LossStrategy(combine_loss_sum, 'L1 + L2 > LT', 'suml1l2_loss'),
                     's4': LossStrategy(combine_loss_max, 'L1 | L2 > LT', 'maxl1l2_loss'),
                     's5': LossStrategy(combine_loss_min, 'L1 & L2 > LT', 'minl1l2_loss')
                 })