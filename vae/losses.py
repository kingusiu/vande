import tensorflow as tf

import numpy as np
import config.config as co

# ********************************************************
#                   training losses
# ********************************************************

### KL
@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_sum(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)

# wrapper for mse loss to pass as reco loss
def mse_loss(inputs, outputs):
    return tf.keras.losses.MeanSquaredError()(inputs, outputs)


### MSE + KL

def make_mse_kl_loss(z_mean, z_log_var, beta):

    mse_loss = tf.keras.losses.MeanSquaredError()
    kl_loss = make_kl_loss(z_mean, z_log_var)
    # multiplying back by N because input is so sparse -> average error very small 
    @tf.function
    def mse_kl_loss(inputs, outputs):
        #input_size = inputs.get_shape().as_list()
        return mse_loss(inputs, outputs) + beta * kl_loss(inputs, outputs)

    return mse_kl_loss

### EXPONENTIAL

def log_k_loss(inputs,outputs):
    log_k = tf.log(outputs)
    return tf.reduce_sum( log_k )


def k_times_x_loss(inputs,outputs):
    return tf.reduce_sum(inputs*outputs) # tf * operator = element wise


# this loss is PER SAMPLE, inputs = 32x32 pixel array, outputs = 32x32 estimates of k of k * e ^-kx
def exponential_prob_loss(inputs,outputs):
    # compute negative log likelihood of probability of inputs under learned model
    return k_times_x_loss(inputs,outputs) - log_k_loss(inputs,outputs)


def exponential_prob_kl_loss( z_mean, z_log_var ):

    def loss( inputs, outputs ):
        return exponential_prob_loss(inputs,outputs) + co.config['beta'] * kl_loss( z_mean, z_log_var )

    return loss


### 3D LOSS
@tf.function
def threeD_loss(inputs, outputs): #[batch_size x 100 x 3]
    expand_inputs = tf.expand_dims(inputs, 2) # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_outputs = tf.expand_dims(outputs, 1) # add broadcasting dim [batch_size x 1 x 100 x 3]
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100] where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = tf.math.reduce_sum(tf.math.squared_difference(expand_inputs, expand_outputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = tf.math.reduce_min(distances,1)
    min_dist_to_outputs = tf.math.reduce_min(distances,2)
    return tf.math.reduce_sum(min_dist_to_inputs, 1) + tf.math.reduce_sum(min_dist_to_outputs, 1)


def make_threeD_kl_loss(z_mean, z_log_var, beta):
    kl_loss = make_kl_loss(z_mean, z_log_var)
    def threeD_kl_loss(inputs, outputs):
        return threeD_loss(inputs, outputs) + beta * kl_loss(inputs, outputs)
    return threeD_kl_loss



### L2 weight norm loss
@tf.function
def l2_regularize(weights):
    ''' squared L2 norm of kernel weights '''
    # get kernel weights (discard bias) for each layer, squared and summed
    kernel_weights = [tf.reduce_sum(tf.square(w)) for w in weights if len(w.shape) > 1]
    # return overall sum
    return tf.reduce_sum(kernel_weights)


# ********************************************************
#                   manual analysis losses
# ********************************************************

def mse_loss_manual(inputs, outputs):
    inputs = inputs.reshape(inputs.shape[0],-1) # flatten to tensor of n events, each of size 1024 (32x32)
    outputs = outputs.reshape(outputs.shape[0],-1)
    reconstruction_loss = np.mean( np.square(outputs-inputs), axis=-1)
    return np.array(reconstruction_loss)


def kl_loss_manual(z_mean,z_log_var):
    kl = 1. + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = -0.5 * np.sum(kl, axis=-1)
    return np.array(kl)


# compute losses manually given original data (inputs), predicted data (outputs) and bg_z_mean and z_log var
def compute_loss_of_prediction_mse_kl(input, predicted, z_mean, z_log_var, beta):
    reco_losses = mse_loss_manual(input, predicted)
    kl_losses = kl_loss_manual(z_mean, z_log_var)
    total_losses = reco_losses + beta * kl_losses # custom loss = mse_loss + l * kl_loss
    return [ total_losses, reco_losses, kl_losses ]


def threeD_loss_manual(inputs, outputs):
    distances = np.sum(np.subtract(inputs[:,:,np.newaxis,:],outputs[:,np.newaxis,:,:])**2, axis=-1)
    min_dist_to_inputs = np.min(distances,axis=1)
    min_dist_to_outputs = np.min(distances,axis=2)
    return np.sum(min_dist_to_inputs,axis=1) + np.sum(min_dist_to_outputs,axis=1)


def compute_loss_of_prediction_3D_kl(inputs, predicted, z_mean, z_log_var, beta):
    reco_losses = threeD_loss_manual(inputs, predicted)
    kl_losses = kl_loss_manual(z_mean, z_log_var)
    total_losses = reco_losses + beta * kl_losses
    return [total_losses, reco_losses, kl_losses]
