import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
import vae.layers as lays
import numpy as np
import tensorflow as tf



class StdNormalizationTest(unittest.TestCase):

    def setUp(self):
        self.in_uni = np.random.uniform(size=[5,4,3])
        self.in_exp = np.random.exponential(10, size=[6,7,4])
        self.in_nor = np.random.normal(30, 2, size=[10,7,2])


    # test normalization with uniform input
    def test_uniform_std_normalize(self):

        for inputs in (self.in_uni, self.in_exp, self.in_nor):
            mean = np.nanmean(inputs, axis=(0,1))
            std = np.nanstd(inputs, axis=(0,1))
            # import ipdb; ipdb.set_trace()
            normalize = lays.StdNormalization(mean, std)
            
            unnormalize = lays.StdUnnormalization(mean, std)
            x = normalize(inputs)
            y = unnormalize(x)

            # check all features normalized to gaussian with mu=0 and std=1
            for i in range(inputs.shape[-1]):
                self.assertAlmostEqual(np.mean(x[:,:,i]), 0.0, places=5)
                self.assertAlmostEqual(np.std(x[:,:,i]), 1.0, places=5)
            np.testing.assert_almost_equal(inputs, y.numpy(), 5)

    def test_normalize_not_trainable(self):
        mean = np.nanmean(self.in_uni, axis=(0,1))
        std = np.nanstd(self.in_uni, axis=(0,1))
        # import ipdb; ipdb.set_trace()
        normalize = lays.StdNormalization(mean, std)
        unnormalize = lays.StdUnnormalization(mean, std)
        self.assertFalse(normalize.trainable)        
        self.assertFalse(unnormalize.trainable)        



class SamplingLayerClass(unittest.TestCase):

    def test_normal_output_from_zmu_zstd(self):
        pass

    def test_kl_loss_zero_normal_zmu_zstd(self):
        ''' test that KL loss for sampled data from a N(0,1) dist = 0 '''
        z_mean = np.array(0.) # zero mean
        z_std = np.array(1.) # unit std-dev


if __name__ == '__main__':
    unittest.main()
