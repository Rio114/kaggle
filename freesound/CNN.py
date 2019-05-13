import numpy as np 
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
import gc

from CNN_model import CNN_Model

def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"

    num_freq = 128
    len_div = 256

    model = CNN_Model()

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=64,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            vertical_flip=False)

    batch_size = 32

    epochs = 10
    for n in range(epochs):
        print('epoch No.{}'.format(n))
        print('learning with curated data')

        with open(FOLDER+'preprocessed_dataset/train_arr_0.pickle', 'rb') as f:
            X_train0 = pickle.load(f)
            y_train0 = pickle.load(f)
        with open(PREPROCESS+'train_arr_1.pickle', 'rb') as f:
            X_train1 = pickle.load(f)
            y_train1 = pickle.load(f)
        X_train = np.vstack([X_train0, X_train1])
        y_train = np.vstack([y_train0, y_train1])

        del X_train0, y_train0
        del X_train1, y_train1
        gc.collect()

        with open(PREPROCESS+'val_arr_0.pickle', 'rb') as f:
            X_val0 = pickle.load(f)
            y_val0 = pickle.load(f)
        with open(PREPROCESS+'val_arr_1.pickle', 'rb') as f:
            X_val1 = pickle.load(f)
            y_val1 = pickle.load(f)
        X_val = np.vstack([X_val0, X_val1])
        y_val = np.vstack([y_val0, y_val1])

        del X_val0, y_val0
        del X_val1, y_val1
        gc.collect()

        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
            steps_per_epoch=int(len(y_train)/batch_size),
            epochs=2,
            validation_data=(X_val, y_val))
        del X_train, X_val, y_train, y_val
        gc.collect()

        pick_val = random.sample(range(2),1)[0]
        with open(PREPROCESS+'noisy_val_arr_{}.pickle'.format(pick_val), 'rb') as f:
            X_val = pickle.load(f)
            y_val = pickle.load(f)

        pick = random.sample(range(8),8)
        for i, m in enumerate(pick):
            print('learning with noisy data No.{}'.format(i))
            with open(PREPROCESS+'noisy_train_arr_{}.pickle'.format(m), 'rb') as f:
                X_train = pickle.load(f)
                y_train = pickle.load(f)
            model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size,),
                steps_per_epoch=int(len(y_train)/batch_size),
                epochs=2,
                validation_data=(X_val, y_val))
            del X_train, y_train
            gc.collect()
        model.save(OUTPUT+'20190512_CNN_model.h5', include_optimizer=False)
        del X_val, y_val
        gc.collect()

if __name__ == '__main__':
    main()
