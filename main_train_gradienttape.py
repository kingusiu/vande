import tensorflow as tf
import vae.vae_particle as vap

input_shape = (28, 28)
Parameters = namedtuple('Parameters', 'beta train_total_n valid_total_n batch_n')
params = Parameters(beta=0.01, train_total_n=int(10e5), valid_total_n=int(1e5), batch_n=256) # 'L1L2'

#### get data ####



#### build model ####

vae = vap.VAEparticle(input_shape=input_shape, z_sz=10, filter_ini_n=6, kernel_sz=3)
vae.build(mean_stdev)



