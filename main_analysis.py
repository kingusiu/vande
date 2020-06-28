import analysis_main.main_analysis_losses as al

# ********************************************************
#               runtime params
# ********************************************************

run_n = 0
sample_id_bg = 'qcdSideReco'
sample_id_signal = ['GtoWW25brReco','GtoWW35naReco']

# analyze losses
al.analyze_losses(run_n,sample_id_bg, sample_id_signal)




