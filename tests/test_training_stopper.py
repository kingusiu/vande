import unittest
import tensorflow as tf
import vande.training as train

class TrainStopperTest(unittest.TestCase):


    def test_stop_training_after_patience_num_losses(self):

        lr = 0.001
        patience = 4
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
        stopper = train.Stopper(optimizer, min_delta=0.01, patience=patience, max_lr_decay=10)

        same_loss_val = 1.2
        losses = np.array([])
        epoch_curr = 0

        for epoch in range(patience-1):
            # add loss
            losses = np.append(losses, same_loss_val)
            # check first #patience-1 epochs do not stop training
            self.assertFalse(stopper.check_stop_training(losses))
            # check first #patience-1 epochs do not alter learning rate
            self.assertEqual(optimizer.learning_rate.numpy(), lr)
            epoch_curr += 1

        for epoch in range(100):
            pass


    def test_never_stop_training_with_changing_loss(self):

        lr = 0.001
        patience = 4
        min_delta = 0.01
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
        stopper = train.Stopper(optimizer, min_delta=min_delta, patience=patience, max_lr_decay=10)

        losses = np.array([1.2])

        for epoch in range(100):
            # check never stop training
            self.assertFalse(stopper.check_stop_training(losses))
            # check never change learning rate
            self.assertEqual(optimizer.learning_rate.numpy(), lr)
            # add new loss that is 2 times larger than min_delta
            losses = np.append(losses, losses[-1]*(1.+min_delta*2))


if __name__ == '__main__':
    unittest.main()
