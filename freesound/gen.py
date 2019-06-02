import numpy as np
from random import shuffle
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

class Generator(object):
    def __init__(self, batch_size, path_prefix, train_keys, val_keys, target_dict, img_size):
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.target_dict = target_dict
        self.image_size = img_size  # (len_div, num_freq)

    def file2np(self, fname):
        n_mels=self.image_size[1]
        y, sr = librosa.load(self.path_prefix + fname)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max) # [-80, 0.0]
        X = (log_S + 80) / 80 # [0.0, 1.0]
        return X

    def generate(self, train=True):
        img_size = self.image_size
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = self.file2np(img_path)

    

                y = self.bbox_util.assign_boxes(y)
                inputs.append(img/255)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets