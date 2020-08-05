import os

import POfAH.sample_dict as sd
import analysis.analysis_jet_feature as ajf
import POfAH.util.experiment as ex
import POfAH.jet_sample as js
import util.event_sample as es
import util.plotting as up
import POfAH.util.sample_factory as sf


run_n = 101
data_sample = 'particle-local'

experiment = ex.Experiment(run_n).setup(fig_dir=True)
paths = sf.SamplePathFactory(experiment, data_sample)


# ********************************************************
#       read in training data ( events )
# ********************************************************

qcd_sample = js.JetSample.from_input_file('qcdSide',paths.qcd_file_path)
scikit_qcd_sample = ajf.dijet_sample_from_dijet_sample(qcd_sample)
ptjj = [j.pt for j in scikit_qcd_sample]

up.plot_hist(ptjj, title='ptjj', plot_name='ptjj_dist', fig_dir=experiment.fig_dir, xlim=(-1,10), bins=3000)



# scikit dijet sample from particle sample
event_sample = es.EventSample.from_input_file('GtoWW30br','../data/events/RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_concat_10K.h5')
j1_particles, j2_particles = event_sample.get_particles()
j1_scikit = ajf.jet_sample_from_particle_sample(j1_particles)
j2_scikit = ajf.jet_sample_from_particle_sample(j2_particles)
jj_scikit = [j1+j2 for j1,j2 in zip(j1_scikit,j2_scikit)]
mjj_scikit = [j.mass for j in jj_scikit]
ptjj_scikit = [j.pt for j in jj_scikit]

# original dijet sample
jet_sample = js.JetSample.from_input_file('GtoWW30br','../data/events/RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_concat_10K.h5')
#scikit dijet sample from original dijet sample
jet_sample_scikit = ajf.dijet_sample_from_dijet_sample(jet_sample)
mjj_scikit_from_orig = [j.mass for j in jet_sample_scikit]
ptjj_scikit_from_orig = [j.pt for j in jet_sample_scikit]

up.plot_hist(mjj_scikit,title='mjj scikit',plot_name='hist_mjj_scikit',fig_dir=experiment.fig_dir)
up.plot_hist(jet_sample['mJJ'],title='mjj original',plot_name='hist_mjj_original',fig_dir=experiment.fig_dir)
up.plot_hist(mjj_scikit_from_orig,title='mjj scikit from original',plot_name='hist_mjj_scikit_from_original',fig_dir=experiment.fig_dir)

up.plot_hist(ptjj_scikit,title='ptjj scikit',plot_name='hist_ptjj_scikit',fig_dir=experiment.fig_dir)
up.plot_hist(ptjj_scikit_from_orig,title='ptjj scikit from original',plot_name='hist_ptjj_scikit_from_original',fig_dir=experiment.fig_dir)

exit()


# select model
experiment = ex.Experiment(run_n=45)

# read in original and reconstructed datasets

sample_original_id = 'GtoWW35br'
jet_sample_original = js.JetSample.from_input_file(sample_original_id, os.path.join(sd.base_dir_events, sd.file_names[sample_original_id]+'_mjj_cut_concat_200K.h5'))

sample_reco_id = 'GtoWW35brReco'
evt_sample_reco = es.EventSample.from_input_file(sample_reco_id,os.path.join(experiment.result_dir, sd.file_names[sample_reco_id]+'.h5'))
j1_reco_particles, j2_reco_particles = evt_sample_reco.get_particles()

# compute jets from particles
j1_reco = ajf.jet_sample_from_particle_sample(j1_reco_particles)
j2_reco = ajf.jet_sample_from_particle_sample(j2_reco_particles)
jj_reco = [j1+j2 for j1,j2 in zip(j1_reco,j2_reco)]
# get dijet mass and dijet pt
m_jj_reco = [ event_jj.mass for event_jj in jj_reco]
pt_jj_reco = [ event_jj.pt for event_jj in jj_reco]

# compute dijet pt for original data sample
jj_orig = ajf.dijet_sample_from_dijet_sample(jet_sample_original)
m_jj_orig_scikit = [event.mass for event in jj_orig]
m_jj_orig = jet_sample_original['mJJ']
pt_jj_orig_scikit = [event.pt for event in jj_orig]

up.plot_hist([m_jj_orig,m_jj_orig_scikit,m_jj_reco],xlabel='mJJ',title='mJJ distribution',plot_name='hist_jet_feature_mjj',legend=['orig','orig scikit','reco'],fig_dir=experiment.fig_dir)
up.plot_hist([pt_jj_orig_scikit,pt_jj_reco],xlabel='ptJJ',title='ptJJ distribution',plot_name='hist_jet_feature_ptjj',legend=['orig scikit','reco'],fig_dir=experiment.fig_dir)
