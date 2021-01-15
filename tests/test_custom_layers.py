import setGPU
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
            normalize = lays.StdNormalize(mean, std)
            
            unnormalize = lays.StdUnnormalize(mean, std)
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
        normalize = lays.StdNormalize(mean, std)
        unnormalize = lays.StdUnnormalize(mean, std)
        self.assertFalse(normalize.trainable)        
        self.assertFalse(unnormalize.trainable)        


if __name__ == '__main__':
    unittest.main()
