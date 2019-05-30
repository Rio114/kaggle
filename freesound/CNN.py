import numpy as np 
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import optimizers
import gc

from CNN_model import CNN_Model
from keras import models
from lwlrap import tf_lwlrap


def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"

    num_freq = 128
    len_div = 256

    # model = CNN_Model()
    model = load_model(OUTPUT+'20190529_BiLSTM_model.h5', compile=False)
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf_lwlrap])

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

    epochs = 100
    batch_size = 32

    for e in range(epochs):
        print('epoch No.', e)
        print('training with curated data...')
        with open(PREPROCESS+'train_curated_0.pickle', 'rb') as f:
            X_train = pickle.load(f)
            y_train = pickle.load(f)
        X_train = X_train.reshape(-1, len_div, num_freq, 1)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
            steps_per_epoch=len(y_train)//batch_size,
            epochs=2,
            validation_data=(X_val, y_val))
        del X_train, y_train
        gc.collect()

        for i in range(5):
            print('training with noisy data No.{}...'.format(i))
            with open(PREPROCESS+'train_noisy_{}.pickle'.format(i), 'rb') as f:
                X_train = pickle.load(f)
                y_train = pickle.load(f)
            X_train = X_train.reshape(-1, len_div, num_freq, 1)
            model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
                steps_per_epoch=len(y_train)//batch_size,
                epochs=2,
                validation_data=(X_val, y_val))
            del X_train, y_train
            gc.collect()
        model.save(OUTPUT+'20190529_BiLSTM_model.h5', include_optimizer=False)
        
if __name__ == '__main__':
    main()
