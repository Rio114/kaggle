from keras.models import Model
from keras.layers import Conv2D, Embedding, MaxPooling2D, Input, Dense, Dot
from keras.layers import Flatten, Reshape, Activation ,Concatenate, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras import optimizers
import tensorflow as tf
from keras.layers.recurrent import LSTM

from lwlrap import tf_lwlrap
import numpy as np

class LSTM_attention():
    def __init__(self, num_freq = 128, len_div = 256, num_hidden=100):
        self.num_freq = num_freq
        self.len_div = len_div
        self.num_hidden = num_hidden
        
    def LSTM(self):

        self.inputs = Input(shape=(self.len_div, self.num_freq, 1), name='input')
        self.resh = Reshape([self.len_div, self.num_freq])(self.inputs)
        
        self.lstm_for = LSTM(self.num_hidden, return_sequences=False, go_backwards=False, name='LSTM_for')(self.resh)
        self.lstm_inv = LSTM(self.num_hidden, return_sequences=False, go_backwards=True, name='LSTM_inv')(self.resh)
        self.conc = Concatenate(axis=2, name='concat')([self.lstm_for, self.lstm_inv])
        self.dens = Dense(self.num_freq, name='dense_for_inv')(self.conc)        
        self.drop = Dropout(rate=0.05, name='drop')(self.dens)
        self.dens1 = Dense(80, name='dense1')(self.drop)
        self.norm = BatchNormalization(axis=-1, name='norm')(self.dens1)
        self.flat = Flatten(name='flatten')(self.norm)
        self.dens2 = Dense(80, name='dense2')(self.flat)
        self.pred = Activation('softmax',name='pred')(self.dens2)

        adam = optimizers.Adam(lr=0.0001)
        model = Model(inputs=self.inputs, outputs=self.pred)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf_lwlrap])
        return model