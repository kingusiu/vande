import numpy as np
import vae.losses as lo

def test_manual_3d_loss():
	a = np.asarray(np.asarray(range(10,50,10)),
		np.asarray(range(20,60,10)),
		np.asarray(range(30,70,10)))
	print(a)

test_manual_3d_loss()