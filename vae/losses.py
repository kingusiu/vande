import numpy as np
from keras.losses import mse
import keras.backend as K
from config import *

# ********************************************************
#                   training losses
# ********************************************************

### KL

def kl_loss( z_mean, z_log_var ):
    kl = 1. + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl = K.sum(kl, axis=-1)
    kl *= -0.5
    return kl

# metric is called with y_true and y_pred, so need function closure to evaluate z_mean and z_log_var instead
def kl_loss_for_metric( z_mean, z_log_var ):

    def loss( inputs, outputs ):
        #return config['beta'] * kl_loss( z_mean, z_log_var ) # ignore inputs / outputs arguments passed in by keras
        return kl_loss( z_mean, z_log_var ) # ignore inputs / outputs arguments passed in by keras

    return loss

### MSE

def mse_loss( inputs, outputs ):
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= config['image_size'] * config['image_size']  # mse returns mean sqr err, so multiply by n
    return reconstruction_loss  # returns scalar (one for each input sample)


def mse_kl_loss( z_mean, z_log_var ):

    def loss( inputs, outputs ):
        return mse_loss( inputs, outputs ) + config['beta'] * kl_loss( z_mean, z_log_var )

    return loss

### EXPONENTIAL

def log_k_loss(inputs,outputs):
    log_k = K.log(K.flatten(outputs))
    return K.sum( log_k )


def k_times_x_loss(inputs,outputs):
    return K.sum(K.flatten(inputs)*K.flatten(outputs)) # keras * operator = element wise


# this loss is PER SAMPLE, inputs = 32x32 pixel array, outputs = 32x32 estimates of k of k * e ^-kx
def exponential_prob_loss(inputs,outputs):
    # compute negative log likelihood of probability of inputs under learned model
    return k_times_x_loss(inputs,outputs) - log_k_loss(inputs,outputs)


def exponential_prob_kl_loss( z_mean, z_log_var ):

    def loss( inputs, outputs ):
        return exponential_prob_loss(inputs,outputs) + config['beta'] * kl_loss( z_mean, z_log_var )

    return loss

# ********************************************************
#                   manual analysis losses
# ********************************************************

def mse_loss_manual(inputs,outputs):
    inputs = inputs.reshape(inputs.shape[0],-1) # flatten to tensor of n events, each of size 1024 (32x32)
    outputs = outputs.reshape(outputs.shape[0],-1)
    reconstruction_loss = np.mean( np.square(outputs-inputs), axis=-1)
    reconstruction_loss *= config['image_size'] * config['image_size'] # mse returns mean sqr err, so multiply by n
    return np.array(reconstruction_loss)


def kl_loss_manual(z_mean,z_log_var):
    kl = 1. + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = np.sum(kl, axis=-1)
    kl *= -0.5
    return np.array(kl)


# compute losses manually given original data (inputs), predicted data (outputs) and bg_z_mean and z_log var
def compute_loss_of_prediction( input, predicted, z_mean, z_log_var ):
    reco_losses = mse_loss_manual(input, predicted)
    kl_losses = kl_loss_manual(z_mean, z_log_var)
    total_losses = reco_losses + config['beta'] * kl_losses # custom loss = mse_loss + l * kl_loss
    return [ total_losses, reco_losses, kl_losses ]
