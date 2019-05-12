
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

def CNN_LSTM_Model(num_freq = 128, len_div = 256):
    n_hidden = 300
    n_filter = 8
    
    inputs = Input(shape=(num_freq,len_div,1 ), name='input')
    
    conv_list = []

    conv1 = Conv2D(n_filter, (17, 1),activation='relu',padding='same',name='conv1')(inputs)
    pool1 = MaxPooling2D((17, 1),strides=(4, 1),padding='same',name='pool1')(conv1)
    norm1 = BatchNormalization(axis=-1, name='norm1')(pool1)
    reshape1 = Reshape([-1, len_div])(norm1)
    conv_list.append(reshape1)

    conv2 = Conv2D(n_filter, (11, 1),activation='relu',padding='same',name='conv2')(inputs)
    pool2 = MaxPooling2D((11, 1),strides=(2, 1),padding='same',name='pool2')(conv2)
    norm2 = BatchNormalization(axis=-1, name='norm2')(pool2)
    reshape2 = Reshape([-1, len_div])(norm2)
    conv_list.append(reshape2)

    conv3 = Conv2D(n_filter, (7, 1),activation='relu',padding='same',name='conv3')(inputs)
    pool3 = MaxPooling2D((7, 1),strides=(4, 1),padding='same',name='pool3')(conv3)
    norm3 = BatchNormalization(axis=-1, name='norm3')(pool3)
    reshape3 = Reshape([-1, len_div])(norm3)
    conv_list.append(reshape3)

    concat = concatenate(conv_list, name='concat', axis=1)

    lstm = LSTM(n_hidden, return_sequences=False, name='LSTM')(concat)
    drop = Dropout(rate=0.05)(lstm)
    dense = Dense(80, name='dense1')(drop)
    pred = Activation('softmax',name='pred')(dense)

    adam = optimizers.Adam(lr=0.0001)

    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf_lwlrap])
    
    return model