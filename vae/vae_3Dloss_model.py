from vae.vae_model import VAE
from vae.losses import *

class VAE_3D( VAE ):

    def __init__(self,**kwargs):
        super(VAE_3D,self).__init__(**kwargs)
        self.input_shape = (100,3,1)

    def load( self, run = 0 ):
        pass
        # todo: load with custom 3d loss name

    def compile(self,model):
        model.compile(optimizer='adam', loss=threeD_kl_loss(self.z_mean, self.z_log_var), metrics=[threeD_loss,kl_loss_for_metric(self.z_mean,self.z_log_var)])  # , metrics=loss_metrics monitor mse and kl terms of loss 'rmsprop'
