"""
yolov7.py
Aimbot YoloV7-tiny-320-800 pre/post processing
---------
Copyright (c) 2024 MemryX Inc.


YoloV7 Weights
--------------
Copyright (c) 2023 Chien-Yao Wang, Alexey Bochkovskiy, Alexey anHong-Yuan Mark Liao


GPL v3 License

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see http://www.gnu.org/licenses/gpl-3.0

"""

############################################################################### ####################

# Imports
import onnx, onnxruntime
onnxruntime.set_default_logger_severity(3)

import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

###################################################################################################
###################################################################################################
###################################################################################################

class YoloV7Tiny:
    """
    A helper class to run YOLOv7 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV7Tiny-320_800'
        self.input_size = (320,800,3)
        self.output_size = [(40,100,255),(20,50,255),(10,25,255)]

        # don't let ORT go wild with threads
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        # post model is so light we just always use CPU
        self.post_model = onnxruntime.InferenceSession('../models/yolov7-tiny_320_800.post.onnx', sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

###################################################################################################
    def preprocess(self, img):
        """
        YOLOv7 Pre-proccessing.
        """
        h0, w0 = img.shape[:2] # input size

        # resize img to model's img_size
        hr = self.input_size[0] / h0
        wr = self.input_size[1] / w0
        if (hr != 1) or (wr != 1):
            interp = cv2.INTER_AREA if ((wr < 1) or (hr < 1)) else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * wr), int(h0 * hr)), interpolation=interp)
        # resized w/h
        h, w = img.shape[:2]

        # letterbox to maintain aspect ratio (320x800)
        img, ratio_w, ratio_h, dwdh = self._letterbox(img, new_shape=self.input_size, auto=False)
        img = img.astype(np.float32)
        img /= 255.0 # Scale

        shapes = (h0, w0), ((h / h0, w / w0), dwdh)

        if not self.stream_mode:
            self.ratio_w = wr
            self.ratio_h = hr
            self.pad = dwdh

        return img

###################################################################################################
    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32):
        """
        A letterbox function.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        hr = new_shape[0] / shape[0]
        wr = new_shape[1] / shape[1]
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            hr = min(hr, 1.0)
            wr = min(wr, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * wr)), int(round(shape[0] * hr))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, wr, hr, (dw, dh)

###################################################################################################
    def postprocess(self, fmap, ch_first=False):
        """
        YOLOv7 Post-proccessing.
        """
        if len(fmap[0].shape) == 3:
            _fmap = [f[None,:,:,:] for f in fmap]
        else:
            _fmap = fmap

        if ch_first:
            post_input = {
                self.post_model.get_inputs()[0].name: _fmap[0],
                self.post_model.get_inputs()[1].name: _fmap[1],
                self.post_model.get_inputs()[2].name: _fmap[2]
            }
        else:
            post_input = {
                self.post_model.get_inputs()[0].name: np.moveaxis(_fmap[0],-1,1),
                self.post_model.get_inputs()[1].name: np.moveaxis(_fmap[1],-1,1),
                self.post_model.get_inputs()[2].name: np.moveaxis(_fmap[2],-1,1)
            }
        post_output = self.post_model.run(None, post_input)[0]

        dets = []
        # run post process model
        for i, arr in enumerate(post_output):
            if arr[6] < 0.4:
                continue
            unpad = arr[1:5]-np.array([self.pad[0], self.pad[1], self.pad[0], self.pad[1]])
            x1 = int(unpad[0]/self.ratio_w)
            y1 = int(unpad[1]/self.ratio_h)
            x2 = int(unpad[2]/self.ratio_w)
            y2 = int(unpad[3]/self.ratio_h)
            det = {}
            det['bbox'] = (x1,y1,x2,y2)
            det['class'] = COCO_CLASSES[int(arr[5])]
            det['class_idx'] = int(arr[5])
            det['score'] = arr[6]
            dets.append(det)
        return dets

###################################################################################################
if __name__=="__main__":
    pass

# eof
