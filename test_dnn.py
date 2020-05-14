import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.losses import mean_squared_error

from input_data_reader import *

class TestDNN():

    def build(self):
        input = Input(shape=(32, 32, 1))
        hidden = Conv2D(filters=3, kernel_size=(3, 3), activation='relu')(input)
        hidden = Flatten()(hidden)
        hidden = Dense(10, activation='relu')(hidden)
        output = Dense(1, activation='relu')(hidden)

        model = Model(input, output)
        model.summary()
        model.compile(optimizer='sgd', loss=mean_squared_error)

        self.model = model

    def fit(self,x,y):
        self.model.fit(x,y,batch_size=20,epochs=3,verbose=2)


dnn = TestDNN()
dnn.build()

qcd_img_j1, qcd_img_j2, sig_img_j1, sig_img_j2 = CaseInputDataReader( './data/BB_images_batch10_subset1000.h5' ).read_images( )
#x = np.random.randint(10,size=(60,32,32,1))
x = qcd_img_j1[:60]
y = np.random.randint(10,size=60)

dnn.fit(x,y)
