import numpy as np 
import keras.backend as K
from keras.engine.topology import Layer

class DefaultBox(Layer):
    def __init__(self, img_size, min_size, max_size, aspect_ratios, variances, **kwargs):
        self.img_size = img_size
        if min_size < 0:
            raise Exception('min_size must be positive')
        self.min_size = min_size
        if max_size < min_size:
            raise Exception('max_size must be grater than min_size')
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        
        super(DefaultBox, self).__init__(**kwargs)

    def compute_output_shape(self, x):
        num_priors_ = len(self.aspect_ratios)
        layer_height = x[1]
        layer_width = x[2]
        num_boxes = num_priors_ * layer_width * layer_height
        return (x[0], num_boxes, 8)

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)

        layer_height = input_shape[1]
        layer_width = input_shape[2]
        
        img_height = self.img_size[0]
        img_width = self.img_size[1]

        box_heights = []
        box_widths = []

        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_heights.append(self.min_size)
                box_widths.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_heights.append(np.sqrt(self.min_size * self.max_size))
                box_widths.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_heights.append(self.min_size / np.sqrt(ar))
                box_widths.append(self.min_size * np.sqrt(ar))

        box_heights = 0.5 * np.array(box_heights)
        box_widths = 0.5 * np.array(box_widths)

        step_y = img_height / layer_height       
        step_x = img_width / layer_width
        lin_y = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        lin_x = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        centers_y, centers_x = np.meshgrid(lin_y, lin_x)
        centers_y = centers_y.reshape(-1, 1)
        centers_x = centers_x.reshape(-1, 1)

        num_defaults = len(self.aspect_ratios)
        default_boxes = np.concatenate((centers_y, centers_x), axis=1)
        default_boxes = np.tile(default_boxes, (1, 2 * num_defaults))
        default_boxes[:, ::4] -= box_heights
        default_boxes[:, 1::4] -= box_widths
        default_boxes[:, 2::4] += box_heights
        default_boxes[:, 3::4] += box_widths
        default_boxes[:, ::2] /= img_height
        default_boxes[:, 1::2] /= img_width
        default_boxes = default_boxes.reshape(-1, 4)
        num_boxes = len(default_boxes)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        default_boxes = np.concatenate((default_boxes, variances), axis=1)
        default_boxes_tensor = K.expand_dims(K.variable(default_boxes), 0)

        pattern = [K.shape(x)[0], 1, 1]
        default_boxes_tensor = K.tile(default_boxes_tensor, pattern)

        return default_boxes_tensor
