import numpy as np
import pandas as pd
import datetime as dt
from os import listdir
import imp
import Annex
imp.reload(Annex)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,SimpleRNN, Lambda,GRU,AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D,LSTM,LocallyConnected2D,Convolution2D,Reshape,Conv1D
from keras.utils import np_utils
from keras.optimizers import RMSprop,Nadam,Adam,SGD
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from sklearn.model_selection import GridSearchCV

print(keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

X_train,X_test,Y_train,Y_test,X,Y,scaler=Annex.get_data_raw(scale=True, 
                                                        add_dummies=True,
                                                        var_dummies=['insee','ddH10_rose4'],
                                                        TrainTestSplit=True,
                                                        sz_test=0.1,
                                                        impute_method='drop',
                                                        convert_month2int=True,
                                                        date_method='drop')
                                                        
def baseline_model():    
    # create model  
    model = Sequential()  
    model.add(Dense(200, input_shape=(38,), kernel_initializer='normal', activation='relu'))  
    model.add(Dense(200, kernel_initializer='normal', activation='relu'),) 
    model.add(Activation("relu"))
    model.add(Dense(1, kernel_initializer='normal'))  
    # Compile model  
    model.compile(loss='mean_squared_error', optimizer='adam')  
    return model  

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset

nb_epoch=15
batch_size=15

ts = time.time()
nnet = KerasRegressor(build_fn=baseline_model, batch_size=batch_size, verbose=1)

nnet.fit(X_train,Y_train,epochs=nb_epoch)
score = nnet.score(X_test, Y_test)
ypred = nnet.predict(X_test)
te = time.time()

print("Score : %f, time running : %d secondes" %(score, te-ts))