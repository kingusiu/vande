from io.input_data_reader import *
from util_plotting import *

# read in result file
results = InputDataReader( './results/AtoHZtoZZZ_results.h5' ).read_events_results( )
print(type(results))

# ****************************************
#           plot losses
# ****************************************

# j1 loss vs j2 loss
plot_hist_2d( results['j1TotalLoss'], results['j2TotalLoss'], 'L_j1', 'L_j2', 'loss jet1 vs jet2', 'Lj1_vs_Lj2')
