from vae_model import *
from input_data_reader import *

vae = VAE()
vae.build()

qcd_img_j1, qcd_img_j2, sig_img_j1, sig_img_j2 = CaseInputDataReader( './data/BB_images_batch10_subset1000.h5' ).read_images( )
#qcd_img_j1, qcd_img_j2 = InputDataReader( './data/images/background_small_img_stdnormal_bin32.h5' ).read_images( )
training_img = np.vstack([qcd_img_j1,qcd_img_j2])
np.random.shuffle( training_img )
history = vae.fit( training_img, training_img, epochs=100, verbose=2 )

vae.save_model( run=1 )

print(history.history.keys())
