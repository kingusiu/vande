import operator
import numpy as np

def filter_arrays_on_value(*arrays, filter_arr, filter_val, comp=operator.gt):
    idx_after_cut = comp(filter_arr,filter_val)
    print('{0} events passed mass cut at {1}'.format(sum(idx_after_cut), filter_val))
    return [a[idx_after_cut] for a in arrays]


def get_mean_and_std(dat):
	'''
	compute mean and std-dev of each feature (axis 2) of a datasample [N_examples, K_elements, F_features]
	'''
	std = np.nanstd(dat,axis=(0,1))
	mean = np.nanmean(dat,axis=(0,1))
	print('computed mean {} and std-dev {}'.format(mean, std))
	std[std == 0.0] = 1.0 # handle zeros
	return mean, std
