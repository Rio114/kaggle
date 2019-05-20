
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras import optimizers
import tensorflow as tf
from keras.layers.recurrent import LSTM
from lwlrap import tf_lwlrap

def LSTM_Model(num_freq = 128, len_div = 256):
    n_hidden = 300

    inputs = Input(shape=(num_freq,len_div,1), name='input')
    resh = Reshape([num_freq, len_div])(inputs)

    lstm = LSTM(n_hidden, return_sequences=False, name='LSTM')(resh)
    resh1 = Reshape([n_hidden, 1])(lstm)
    lstm2 = LSTM(n_hidden*2, return_sequences=False, name='LSTM')(resh1)
    drop = Dropout(rate=0.05)(lstm)
    dense = Dense(80, name='dense1')(drop)
    pred = Activation('softmax',name='pred')(dense)

    adam = optimizers.Adam(lr=0.0001)

    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf_lwlrap])
    
    return model