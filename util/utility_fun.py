import operator

def filter_arrays_on_value(arrays, filter_array, filter_value, comp=operator.gt):
    idx_after_cut = comp(filter_array,filter_value)
    return [a[idx_after_cut] for a in arrays]