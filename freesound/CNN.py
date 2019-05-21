import numpy as np 
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
import gc

from LSTM_model import LSTM_Model
from keras import models

def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"

    num_freq = 128
    len_div = 256

    model = LSTM_Model()

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=64,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            vertical_flip=False)

    batch_size = 32

    with open(PREPROCESS+'val_arr_0.pickle', 'rb') as f:
        X_val = pickle.load(f)
        y_val = pickle.load(f)
    X_val[:, :, :, np.newaxis]
    X_val = X_val.reshape(-1, num_freq,len_div, 1)
    
    epochs = 30
    for n in range(epochs):
        print('epoch No.{}'.format(n))
        pick = random.sample(range(6),6)
        for i, m in enumerate(pick):
            print('learning data No.{}'.format(i))
            with open(PREPROCESS+'train_arr_{}.pickle'.format(m), 'rb') as f:
                X_train = pickle.load(f)
                y_train = pickle.load(f)
            X_train = X_train.reshape(-1, num_freq,len_div, 1)
            model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
                steps_per_epoch=int(len(y_train)/batch_size),
                epochs=2,
                validation_data=(X_val, y_val))
            del X_train, y_train
            gc.collect()
        model.save(OUTPUT+'20190520_LSTM_model.h5', include_optimizer=False)
        
if __name__ == '__main__':
    main()
