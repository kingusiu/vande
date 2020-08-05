import h5py
import POfAH.util.sample_factory as sf
import POfAH.util.input_data_reader as idr
import POfAH.util.experiment as ex
import POfAH.jet_sample as js


run_n = 0
data_sample = 'img'


experiment = ex.Experiment(run_n).setup(model_dir=True)
paths = sf.SamplePathFactory(experiment, data_sample)

# read signals
sample_ids = ['qcdSide', 'qcdSig', 'GtoWW15na', 'GtoWW15br', 'GtoWW25na', 'GtoWW25br', 'GtoWW35na', 'GtoWW35br', 'GtoWW45na', 'GtoWW45br']

for sample_id in sample_ids:
    print('reading {}'.format(paths.sample_path(sample_id)))
    data_reader = idr.InputDataReader(paths.sample_path(sample_id))
    test_sample = js.JetSample.from_feature_array(sample_id, *data_reader.read_dijet_features())
    print(test_sample.name)
