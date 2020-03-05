from vae_model import *
from input_data_reader import *

vae = VAE()
vae.build()

img_j1, img_j2 = InputDataReader( './data/images/background_small_img_stdnormal_bin32.h5' ).read_images( )
history = vae.fit( img_j1, img_j1, epochs=100, verbose=2 )

vae.save_model()

print(history.history.keys())

