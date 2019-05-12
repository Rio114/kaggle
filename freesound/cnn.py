from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import concatenate

class cnn():
    def __init__(self, img_size=(128,100,1), num_classes=80):
        self.img_size = img_size
        self.num_classes = num_classes
        self.dim_box = 4 #(cx, cy, w, h)

    def model(self):
        """
        """
        net = {}

        # modefied VGG16
        ## Input
        inputs = Input(shape=self.img_size, name='input')

        ## Block 1
        conv1_1 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_1')(inputs)
        conv1_2 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool1')(conv1_2)

        ## Block 2
        conv2_1 = Conv2D(128, (3, 3),activation='relu',padding='same',name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3),activation='relu',padding='same',name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool2')(conv2_2)

        ## Block 3
        conv3_1 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool3')(conv3_3)

        ## Block 4
        conv4_1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool4')(conv4_3)

        ## Block 5
        conv5_1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_1')(pool4)
        conv5_2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool5')(conv5_3)

        ## FC6
        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6),activation='relu', padding='same',name='fc6')(pool5)

        ## FC7
        fc7 = Conv2D(1024, (1, 1), activation='relu',padding='same',name='fc7')(fc6)

        ## Block 6
        conv6_1 = Conv2D(256, (1, 1),activation='relu',padding='same',name='conv6_1')(fc7)
        conv6_2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv6_2')(conv6_1)

        ## Block 7
        conv7_1 = Conv2D(128, (1, 1),activation='relu',padding='same',name='conv7_1')(conv6_2)
        conv7_2 = Conv2D(256, (3, 3),strides=(2,2),activation='relu',padding='valid',name='conv7_2')(conv7_1)

        ## Block 8
        conv8_1 = Conv2D(128, (1, 1),activation='relu',padding='same',name='conv8_1')(conv7_2)
        conv8_2 = Conv2D(256, (3, 3),strides=(2,2),activation='relu',padding='same',name='conv8_2')(conv8_1)

        ## Last Pool
        # pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

        # detectors from layers
        layers = [conv4_3, fc7, conv6_2, conv7_2, conv8_2]
        pred = self.detectors(layers)

return Model(inputs, pred)