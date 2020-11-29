import unittest
import os
import math
import pofah.path_constants.sample_dict_file_parts_input_baby as sdi
import util.data_generator as dage


'''
test obsolete: generator returns only one sample at a time
'''

class DataGeneratorTestCase(unittest.TestCase):

	def setUp(self):
		sample_id = 'qcdSig'
		self.path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir'][sample_id])
		self.batch_sz = 1024
		self.samples_n_total = 116096*2 # 116096 events * 2 jets
		self.generator = dage.DataGenerator(path=self.path, batch_sz=self.batch_sz)
	
	def test_num_events_read(self):
		n_total_read = 0
		for (constituents, _) in self.generator():
			n_batch_read = len(constituents)
			n_total_read += n_batch_read
			# check batch_sz samples in each batch
			self.assertEqual(n_batch_read, self.batch_sz)
		# check all samples read
		self.assertGreaterEqual(n_total_read, self.samples_n_total)
		# check batch padding
		self.assertEqual(math.ceil(self.samples_n_total/self.batch_sz)*self.batch_sz, n_total_read)


	def test_shape_events_read(self):
		for (constituents, _) in self.generator():
			self.assertEqual(constituents.shape, (self.batch_sz, 100, 3))

	def test_num_events_read_max_n(self):
		max_n = 3000
		max_n_generator = dage.DataGenerator(path=self.path, batch_sz=self.batch_sz, max_n=max_n)
		n_total_read = 0
		for (constituents, _) in max_n_generator():
			n_total_read += len(constituents)
		self.assertLess(max_n-n_total_read, self.batch_sz)
		self.assertEqual(n_total_read%self.batch_sz,0)
		self.assertGreater(n_total_read, 0)


if __name__ == '__main__':
	unittest.main()
