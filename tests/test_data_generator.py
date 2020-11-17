import unittest
import os
import pofah.path_constants.sample_dict_file_parts_input_baby as sdi
import util.data_generator as dage


class DataGeneratorTestCase(unittest.TestCase):

	def setUp(self):
		sample_id = 'qcdSig'
		path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir'][sample_id])
		self.batch_sz = 1024
		self.samples_n_total = 116096*2 # 116096 events * 2 jets
		self.generator = dage.DataGenerator(path=path, batch_sz=self.batch_sz)
	
	def test_num_events_read(self):
		n_total_read = 0
		for (constituents, _) in self.generator():
			n_batch_read = len(constituents)
			n_total_read += n_batch_read
			self.assertEqual(n_batch_read, self.batch_sz)
		self.assertGreaterEqual(n_total_read, self.samples_n_total)

	def test_shape_events_read(self):
		for (constituents, _) in self.generator():
			self.assertEqual(constituents.shape, (self.batch_sz, 100, 3))

if __name__ == '__main__':
	unittest.main()
