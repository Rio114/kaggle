import numpy as np 
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
import gc

from Bidir_LSTM_model import LSTM_attention
from keras import models

def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"

    num_freq = 128
    len_div = 256

    model_obj = LSTM_attention()
    model = model_obj.LSTM()

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=64,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            vertical_flip=False)

    with open(PREPROCESS+'val_curated_0.pickle', 'rb') as f:
        X_val = pickle.load(f)
        y_val = pickle.load(f)
    X_val = X_val.reshape(-1, len_div, num_freq, 1)

    epochs = 30
    batch_size = 32
    with open(PREPROCESS+'train_curated_0.pickle', 'rb') as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
    X_train = X_train.reshape(-1, len_div, num_freq, 1)

    for _ in range(epochs):
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
            steps_per_epoch=len(y_train)//batch_size,
            epochs=5,
            validation_data=(X_val, y_val))
        model.save(OUTPUT+'20190522_BiLSTM_model.h5', include_optimizer=False)
        
if __name__ == '__main__':
    main()
