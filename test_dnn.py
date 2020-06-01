import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda
from keras.losses import mean_squared_error
from keras import backend as K

from io.input_data_reader import *

class TestConvDNN():

    def __init__(self):
        self.batch_size = 20

    def custom_activation( self, x ):
        return K.minimum( x, self.activation_cut )

    def build(self):
        input = Input(shape=(32, 32, 1))
        hidden = Conv2D(filters=3, kernel_size=(3, 3), activation='relu')(input)
        hidden = Flatten()(hidden)
        hidden = Dense(10, activation='relu')(hidden)
        hidden = Dense(1, activation='relu')(hidden)
        output = Dense(1, activation=self.custom_activation)(hidden)

        model = Model(input, output)
        model.summary()
        model.compile(optimizer='sgd', loss=mean_squared_error)

        self.model = model

    def add_layer(self, cut):
        self.activation_cut = K.constant(cut)


    def fit(self,x,y):
        self.model.fit(x,y,batch_size=self.batch_size,epochs=30,verbose=2)

    def predict(self, x):
        return self.model.predict( x, batch_size=self.batch_size)


    def save(self, path):
        self.model.save(path)

    def load(self,path):
        self.model = load_model(path,custom_objects={'custom_activation':self.custom_activation})


dnn = TestConvDNN()
dnn.build()

#qcd_img_j1, qcd_img_j2, sig_img_j1, sig_img_j2 = CaseInputDataReader( './data/BB_images_batch10_subset1000.h5' ).read_images( )
#x = qcd_img_j1[:60]
x = np.random.random(size=(1000,32,32,1))
y = np.random.randint(5,size=1000)

dnn.fit(x,y)

y_pred = dnn.predict( x ).flatten()

dnn.save('models/test_dnn.h5')

dnn2 = TestConvDNN()
dnn2.load('models/test_dnn.h5')

y_pred2 = dnn2.predict(x)

np.testing.assert_allclose( y_pred, y_pred2 )

plt.figure()
plt.plot(y_pred)
plt.plot(y_pred2)
plt.show()
