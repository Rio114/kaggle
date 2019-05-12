import numpy as np 
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator

from CNN_LSTM_model import CNN_LSTM_Model

def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"

    num_freq = 128
    len_div = 256

    model = CNN_LSTM_Model()

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=64,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            vertical_flip=False)

    batch_size = 32

    epochs = 20
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
        
        with open(PREPROCESS+'val_arr_0.pickle', 'rb') as f:
            X_val0 = pickle.load(f)
            y_val0 = pickle.load(f)
        with open(PREPROCESS+'val_arr_1.pickle', 'rb') as f:
            X_val1 = pickle.load(f)
            y_val1 = pickle.load(f)
        X_val = np.vstack([X_val0, X_val1])
        y_val = np.vstack([y_val0, y_val1])
        
        model.fit_generator(datagen.flow(X_train, y_train),
            steps_per_epoch=batch_size,
            epochs=2,
            validation_data=(X_val, y_val))    
        
        pick = random.sample(range(8),8)
        for i, m in enumerate(pick):
            print('learning with noisy data No.{}'.format(i))
            with open(PREPROCESS+'noisy_train_arr_{}.pickle'.format(m), 'rb') as f:
                X_train = pickle.load(f)
                y_train = pickle.load(f)
            pick_val = random.sample(range(2),1)[0]
            with open(PREPROCESS+'noisy_val_arr_{}.pickle'.format(pick_val), 'rb') as f:
                X_val = pickle.load(f)
                y_val = pickle.load(f)
            model.fit_generator(datagen.flow(X_train, y_train),
                steps_per_epoch=batch_size,
                epochs=2,
                validation_data=(X_val, y_val))

    model.save(OUTPUT+'20190512model.h5', include_optimizer=False)

if __name__ == '__main__':
    main()

