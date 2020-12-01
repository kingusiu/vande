import tensorflow as tf
import numpy as np

import pofah.util.event_sample as es




def extract_events(file_path):
	test_sample = es.EventSample.from_input_file('qcdSig', file_path.numpy().decode('utf-8'))
	test_evts_j1, test_evts_j2 = test_sample.get_particles()
	print('{}: {} j1 evts, {} j2 evts'.format(test_sample.name, len(test_evts_j1), len(test_evts_j2)))
	return [test_evts_j1, test_evts_j2]

def map_element_counts(fnames):
  return {'elementA': b'test', 'elementB': 10, 'file': fnames['filenames']}

def test_tf_dataset_for_files(): 

	root_dir = '/eos/home-k/kiwoznia/dev/autoencoder_for_anomaly/convolutional_VAE/data/events/qcd_sqrtshatTeV_13TeV_PU40_parts'

	list_ds = tf.data.Dataset.list_files(root_dir+'/*')

	#list_samples = list_ds.map(extract_events)
	#print(list_samples)

	for file_path in list_ds:
		test_sample = es.EventSample.from_input_file('qcdSig', file_path.numpy().decode('utf-8'))
		test_evts_j1, test_evts_j2 = test_sample.get_particles()
		print('{}: {} j1 evts, {} j2 evts'.format(test_sample.name, len(test_evts_j1), len(test_evts_j2)))


	filelist = ['fileA_6', 'fileB_10', 'fileC_7']


	ds = tf.data.Dataset.from_tensor_slices({'filenames': filelist})
	ds = ds.map(map_func=map_element_counts)
	#element = ds.make_one_shot_iterator().get_next()

	for dat in ds:
		print(dat)


