import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle

from keras import models

def main():
    FOLDER = "../../data_kaggle/freesound/"
    PREPROCESS = FOLDER + "preprocessed_dataset/"
    OUTPUT = FOLDER + "out/"
    INPUT_FOLDER = FOLDER + "input/"

    SAMPLE_SUBMISSION_PATH = INPUT_FOLDER + "sample_submission.csv"
    TEST = INPUT_FOLDER + "test/"
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    num_freq = 128
    len_div = 256

    model = models.load_model(OUTPUT+'20190517_CNN_model.h5', compile=False)
    
    with open(PREPROCESS+'test_arr_0.pickle', 'rb') as f:
        X_test = pickle.load(f)
        file_name = pickle.load(f)

    y_pred = model.predict(X_test.reshape([-1, num_freq, len_div,1]))
    df_pred = pd.DataFrame(y_pred, columns=sample.columns[1:])
    se_fname = pd.Series(file_name, name=sample.columns[0])
    df_temp = pd.concat([se_fname, df_pred], axis=1)
    sub_df = df_temp.groupby(sample.columns[0]).mean()
    sub_df['fname']=sub_df.index

    sub_df.to_csv('submission.csv', index=False)


