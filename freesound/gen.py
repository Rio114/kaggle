import numpy as np
from random import shuffle
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import librosa

class Generator(object):
    def __init__(self, batch_size, df, train_keys, val_keys, target_names, img_size):
        self.batch_size = batch_size
        self.df = df
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.target_names = target_names
        self.image_size = img_size  # (len_div, num_freq)

    def file2np(self, fname):
        n_mels=self.image_size[1]
        y, sr = librosa.load(fname)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max) # [-80, 0.0]
        X = (log_S + 80) / 80 # [0.0, 1.0]
        return X

    def generate(self, train=True):
        img_size = self.image_size
        df = self.df
        while True:
            if train:
                keys = self.train_keys
            else:
                keys = self.val_keys
            inputs = []
            targets = []
            for i, key in enumerate(keys):
                img_path = df.query('fname == "{}"'.format(key))['path'].values[0]
                target = df.query('fname == "{}"'.format(key))[self.target_names].values[0]
                img = self.file2np(img_path)
                
                div = img_size[0] # img_size(len_div, num_freq)
                num_batch = img.shape[1] // div
                rest = img.shape[1] % div

                num_div = img.shape[1] // div
                num_pad = div - img.shape[1] % div
                img_redidual = np.zeros([img_size[1], num_pad])
                img_padded = np.hstack([img, img_redidual])

                img = img_padded[:, :img_size[0]].T.reshape(img_size + (1,))
                inputs.append(img)          
                targets.append(target)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets