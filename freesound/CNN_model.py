
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

def CNN_Model(num_freq = 128, len_div = 256):
    inputs = Input(shape=(len_div,num_freq,1), name='input')

    dense_list = []

    ## Block 1
    conv1 = Conv2D(8, (19, 19),activation='relu',padding='same',name='conv1')(inputs)
    pool1 = MaxPooling2D((19, 19),strides=(2, 2),padding='same',name='pool1')(conv1)
    norm1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm1')(pool1)
    drop1 = Dropout(rate=0.05)(norm1)

    conv1_1 = Conv2D(16, (11, 11),activation='relu',padding='same',name='conv1_1')(drop1)
    pool1_1 = MaxPooling2D((11, 11),strides=(2, 2),padding='same',name='pool1_1')(conv1_1)
    norm1_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm1_1')(pool1_1)
    drop1_1 = Dropout(rate=0.05)(norm1_1)

    conv1_2 = Conv2D(16, (7, 7),activation='relu',padding='same',name='conv1_2')(drop1_1)
    pool1_2 = MaxPooling2D((7, 7),strides=(2, 2),padding='same',name='pool1_2')(conv1_2)
    norm1_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm1_2')(pool1_2)
    drop1_2 = Dropout(rate=0.05)(norm1_2)
                        
    flatten1 = Flatten(name='flatten1')(drop1_2)
    dense1 = Dense(64, name='dense1')(flatten1)
    act1 = Activation('relu',name='act1')(dense1)
    dense_list.append(act1)

    ## Block 2
    conv2 = Conv2D(8, (13, 13),activation='relu',padding='same',name='conv2')(inputs)
    pool2 = MaxPooling2D((13, 13), strides=(2, 2), padding='same',name='pool2')(conv2)
    norm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm2')(pool2)
    drop2 = Dropout(rate=0.05)(norm2)

    conv2_1 = Conv2D(16, (11, 11),activation='relu',padding='same',name='conv2_1')(drop2)
    pool2_1 = MaxPooling2D((11, 11), strides=(2, 2), padding='same',name='pool2_1')(conv2_1)
    norm2_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm2_1')(pool2_1)
    drop2_1 = Dropout(rate=0.05)(norm2_1)

    conv2_2 = Conv2D(16, (7, 7),activation='relu',padding='same',name='conv2_2')(drop2_1)
    pool2_2 = MaxPooling2D((7, 7), strides=(2, 2), padding='same',name='pool2_2')(conv2_2)
    norm2_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm2_2')(pool2_2)
    drop2_2 = Dropout(rate=0.05)(norm2_2)

    flatten2 = Flatten(name='flatten2')(drop2_2)
    dense2 = Dense(64, name='dense2')(flatten2)
    act2 = Activation('relu',name='act2')(dense2)
    dense_list.append(act2)

    ## Block 3
    conv3 = Conv2D(8, (11, 11), activation='relu',padding='same',name='conv3')(inputs)
    pool3 = MaxPooling2D((11, 11), strides=(2, 2), padding='same',name='pool3')(conv3)
    norm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm3')(pool3)
    drop3 = Dropout(rate=0.05)(norm3)

    conv3_1 = Conv2D(16, (7, 7), activation='relu',padding='same',name='conv3_1')(drop3)
    pool3_1 = MaxPooling2D((7, 7), strides=(2, 2), padding='same',name='pool3_1')(conv3_1)
    norm3_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm3_1')(pool3_1)
    drop3_1 = Dropout(rate=0.05)(norm3_1)

    conv3_2 = Conv2D(16, (5, 5), activation='relu',padding='same',name='conv3_2')(drop3_1)
    pool3_2 = MaxPooling2D((5, 5), strides=(2, 2), padding='same',name='pool3_2')(conv3_2)
    norm3_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm3_2')(pool3_2)
    drop3_2 = Dropout(rate=0.05)(norm3)

    flatten3 = Flatten(name='flatten3')(drop3_2)
    dense3 = Dense(64, name='dense3')(flatten3)
    act3 = Activation('relu',name='act3')(dense3)
    dense_list.append(act3)

    ## Block 4
    conv4 = Conv2D(8, (7, 7),activation='relu',padding='same',name='conv4')(inputs)
    pool4 = MaxPooling2D((7, 7), strides=(2, 2), padding='same',name='pool4')(conv4)
    norm4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm4')(pool4)
    drop4 = Dropout(rate=0.05)(norm4)

    conv4_1 = Conv2D(16, (5, 5),activation='relu',padding='same',name='conv4_1')(drop4)
    pool4_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same',name='pool4_1')(conv4_1)
    norm4_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm4_1')(pool4_1)
    drop4_1 = Dropout(rate=0.05)(norm4_1)

    conv4_2 = Conv2D(16, (3, 3),activation='relu',padding='same',name='conv4_2')(drop4_1)
    pool4_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same',name='pool4_2')(conv4_2)
    norm4_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm4_2')(pool4_2)
    drop4_2 = Dropout(rate=0.05)(norm4_2)

    flatten4 = Flatten(name='flatten4')(drop4_2)
    dense4 = Dense(64, name='dense4')(flatten4)
    act4 = Activation('relu',name='act4')(dense4)
    dense_list.append(act4)

    concat = concatenate(dense_list, name='concat', axis=1)

    dense2 = Dense(80, name='dense_all')(concat)
    pred = Activation('softmax',name='pred')(dense2)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf_lwlrap])
    return model