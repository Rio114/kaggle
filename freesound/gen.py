import numpy as np
import pickle

class Generator(object):
    def __init__(self, batch_size, prefix, train_keys, val_keys, img_size):
        self.batch_size = batch_size
        self.prefix = prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.image_size = img_size  # (len_div, num_freq)

    def generate(self, train=True):
        img_size = self.image_size
        while True:
            if train:
                keys = self.train_keys
            else:
                keys = self.val_keys
            inputs = []
            targets = []
            for i, key in enumerate(keys):
                with open(self.prefix+'{}.pickle'.format(key[0]), 'rb') as f:
                            img = pickle.load(f)
                            target = pickle.load(f)
                
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