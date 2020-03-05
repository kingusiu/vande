
def filter_arrays_on_value(arrays, filter_array, filter_value):
    idx_after_cut = filter_array > filter_value
    return [a[idx_after_cut] for a in arrays]