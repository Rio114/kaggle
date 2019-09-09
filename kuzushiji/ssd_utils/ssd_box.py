import numpy as np
import tensorflow as tf

import numpy as np 
import keras.backend as K
from keras.engine.topology import Layer

class DefaultBox(Layer):
    '''
    generate default boxes
    '''
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

class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.
    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [ymin, xmin, ymax, xmax, varyc, varxc, varh, varw].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.
    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.
        # Arguments
            box: Box, numpy tensor of shape (4,). (ymin, xmin, ymax, xmax)
            return_iou: Whether to concat iou to encoded values.
        # Return
            encoded_box: Tensor with encoded box 
                numpy tensor of shape (num_priors, 4 (cy cx, h, w) + int(return_iou)).
        """
        iou = self.iou(box) # arr: ious with all priors
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        # return ious, max and/or over theshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask] 
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center # distance between box and default
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.
        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 4:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.
        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes. (ymin, xmin, ymax, xmax)
            variances: Numpy array of variances.
        # Return
            decode_bbox: Shifted priors.
        """
        prior_height = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_width = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_y = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_x = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        decode_bbox_center_y = mbox_loc[:, 0] * prior_height * variances[:, 0]
        decode_bbox_center_y += prior_center_y
        decode_bbox_center_x = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_x += prior_center_x

        decode_bbox_height = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_height *= prior_height
        decode_bbox_width = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_width *= prior_width

        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width

        decode_bbox = np.concatenate((decode_bbox_ymin[:, None],
                                      decode_bbox_xmin[:, None],
                                      decode_bbox_ymax[:, None],
                                      decode_bbox_xmax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=50,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.
        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.
        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, ymin, xmin, ymax, xmax]
        """
        mbox_loc = predictions[:, :, :4]
        mbox_conf = predictions[:, :, 4:-8]
        mbox_priorbox = predictions[:, :, -8:-4]
        variances = predictions[:, :, -4:]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results
        