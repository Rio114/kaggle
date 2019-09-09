import pandas as pd
import numpy as np
import cv2

def get_gt(df_idx, idx, IMAGES, ohe, CHAR_SIZE=(100, 100)):
    code = df_idx.loc[idx]
    try:
        code_arr = np.array(code['labels'].split(' ')).reshape(-1, 5)
    except:
        return
    df_char = pd.DataFrame(code_arr, columns=['unicode', 'x', 'y', 'w', 'h'])
    df_char[['x','y','w','h']] = df_char[['x','y','w','h']].astype('int')

    path = IMAGES + idx + '.jpg'
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_size = img.shape[:2]

    positions = df_char[['x','y', 'w','h']].values
    char_arr = np.zeros([0, CHAR_SIZE[0], CHAR_SIZE[1]])
    for pos in positions :
        x, y, w, h = pos
        img_char = img_gray[y:y+h, x:x+w]
        w = img_char.shape[1]
        h = img_char.shape[0]
        if w > h:
            delta = (w - h) // 2
            pad = (np.ones([delta, w]) * 255).astype('int')
            img_pad = (255 - np.vstack([pad, img_char, pad])).astype('float32')        
            img_pad_resize = cv2.resize(img_pad, CHAR_SIZE).astype('int')
            char_arr = np.vstack([char_arr, img_pad_resize.reshape([1, CHAR_SIZE[0], CHAR_SIZE[1]])])
        else:
            delta = (h - w) // 2
            pad = (np.ones([h, delta]) * 255).astype('int')
            img_pad = (255 - np.hstack([pad, img_char, pad])).astype('float32')
            img_pad_resize = cv2.resize(img_pad, CHAR_SIZE).astype('int')        
            char_arr = np.vstack([char_arr, img_pad_resize.reshape([1, CHAR_SIZE[0], CHAR_SIZE[1]])])
        
    df_char[['x','w']] = df_char[['x','w']] / img_size[1]
    df_char[['y','h']] = df_char[['y','h']] / img_size[0]
    loc_arr = df_char[['x','y','w','h']].values
    loc_arr[:, 2:4] = loc_arr[:, 2:4] + loc_arr[:, 0:2]
    labels = ohe.transform(df_char['unicode'].values[:, np.newaxis]).toarray()
    gt_char = np.hstack([loc_arr, labels])
    return gt_char  # [xmin, ymin, xmax, ymax, label0, label1, label2,,,]