import numpy as np

def get_test_data_for_vae(*args):
	if not args: args = (100,7,3) # 100 samples, 7 elements, with 3 features each
	return np.random.random(size=args), np.random.randint(5,size=args[0])
