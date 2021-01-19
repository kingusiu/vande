import unittest
import tensorflow as tf
import numpy as np
import vande.training as train


class TrainStopperTest(unittest.TestCase):


    def test_stop_training_after_patience_num_losses(self):

        lr = 0.001
        patience = 4
        max_lr_decay = 3
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
        stopper = train.Stopper(optimizer, min_delta=0.01, patience=patience, max_lr_decay=max_lr_decay)

        same_loss_val = 1.2
        losses = tf.zeros([0])
        epoch_curr = 0

        # check first patience-1 epochs
        for epoch in range(patience-1):
            epoch_curr += 1
            # add loss
            losses = tf.concat([losses, [same_loss_val]], axis=0)
            # check first #patience-1 epochs do not stop training
            self.assertFalse(stopper.check_stop_training(losses))
            # check first #patience-1 epochs do not alter learning rate
            self.assertAlmostEqual(optimizer.learning_rate.numpy(), lr)

        # check epoch #patience++
        lr_curr = lr
        patience_curr = epoch_curr
        for epoch in range(100):
            epoch_curr += 1
            patience_curr += 1
            losses = tf.concat([losses, [same_loss_val]], axis=0)
            stop = stopper.check_stop_training(losses)
            if stop:
                break
            # check learning rate altered every #patience epochs
            if patience_curr < patience: # check lr unaltered
                self.assertAlmostEqual(optimizer.learning_rate.numpy(), lr_curr)
            else: # check lr reduced otherwise
                self.assertLess(optimizer.learning_rate.numpy(), lr_curr)
                lr_curr = optimizer.learning_rate.numpy()
                patience_curr = 0

        # check traning ran for patience * max_lr_decay epochs
        self.assertEqual(epoch_curr, patience * (max_lr_decay+1)) # TODO: check why max_lr_decay+1 ???


    def test_max_lr_decay(self):

        lr = 0.001
        patience = 4
        max_lr_decay = 3
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
        stopper = train.Stopper(optimizer, min_delta=0.01, patience=patience, max_lr_decay=max_lr_decay)

        same_loss_val = 1.2
        losses = tf.zeros([0])

        lr_decayed_n = 0
        lr_curr = lr
        for epoch in range(100):
            losses = tf.concat([losses, [same_loss_val]], axis=0)
            stop = stopper.check_stop_training(losses)
            if lr_curr > optimizer.learning_rate.numpy():
                lr_decayed_n += 1
                lr_curr = optimizer.learning_rate.numpy()
            if stop:
                break

        self.assertEqual(lr_decayed_n, max_lr_decay)    


    def test_never_stop_training_with_changing_loss(self):

        lr = 0.001
        patience = 4
        min_delta = 0.01
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
        stopper = train.Stopper(optimizer, min_delta=min_delta, patience=patience, max_lr_decay=10)

        max_epoch = 100
        losses = tf.constant([1.2])

        for epoch in range(max_epoch):
            # check never stop training
            self.assertFalse(stopper.check_stop_training(losses))
            # check never change learning rate
            self.assertAlmostEqual(optimizer.learning_rate.numpy(), lr)
            # add new loss that is 2 times larger than min_delta
            losses = tf.concat([losses, [losses.numpy()[-1]*(1.+min_delta*2)]], axis=0)

        self.assertEqual(epoch, max_epoch-1)


if __name__ == '__main__':
    unittest.main()
