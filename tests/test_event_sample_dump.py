import os
import POfAH.util.experiment as ex
import util.event_sample as es
import analysis.analysis_constituents as ac
from config import *


experiment = ex.Experiment(run_n=0)

# read in data
test_sample = es.EventSample.from_input_file('RS Graviton WW br 3.0TeV','/home/kinga/mnt/data/events/RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV_concat_10K.h5')
test_evts_j1, test_evts_j2 = test_sample.get_particles()

# setup analysis
particle_analysis = ac.AnalysisConstituents('RS Graviton WW br 3.0TeV', fig_dir='/home/kinga/mnt/fig/run_0/analysis_event')
particle_analysis.analyze( [test_evts_j1, test_evts_j2] )

# dump sample
out_path = os.path.join('/home/kinga/mnt/results/run_0')
test_sample.dump(out_path)

# read back
test_sample = es.EventSample.from_input_file('RS Graviton WW br 3.0TeV dumped', os.path.join(out_path,test_sample.file_name))
test_evts_j1, test_evts_j2 = test_sample.get_particles()

# setup analysis
particle_analysis = ac.AnalysisConstituents('RS Graviton WW br 3.0TeV dumped', fig_dir='/home/kinga/mnt/fig/run_0/analysis_event')
particle_analysis.analyze( [test_evts_j1, test_evts_j2] )