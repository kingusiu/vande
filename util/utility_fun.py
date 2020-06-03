import operator

def filter_arrays_on_value(*arrays, filter_arr, filter_val, comp=operator.gt):
    idx_after_cut = comp(filter_arr,filter_val)
    print('{0} events passed mass cut at {1}'.format(sum(idx_after_cut), filter_val))
    return [a[idx_after_cut] for a in arrays]