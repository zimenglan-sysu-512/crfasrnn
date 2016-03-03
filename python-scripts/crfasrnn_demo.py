# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the 
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on 
top of the Caffe deep learning library.
 
Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor: 
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""

caffe_root = '../caffe-crfrnn/'
import sys
sys.path.insert(0, caffe_root + 'python')

import os
import cv2
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt


MODEL_FILE = '../origin.models/TVG_CRFRNN_COCO_VOC.prototxt'
PRETRAINED = '../origin.models/TVG_CRFRNN_COCO_VOC.caffemodel'
IMAGE_FILE = 'input.jpg'


#caffe.set_mode_gpu()
net = caffe.Segmenter(MODEL_FILE, PRETRAINED, gpu=False)
input_image = 255 * caffe.io.load_image(IMAGE_FILE)


width = input_image.shape[0]
height = input_image.shape[1]
maxDim = max(width,height)

image = PILImage.fromarray(np.uint8(input_image))
image = np.array(image)

# Each row represents one class. For example, [0, 0, 0] means background. 
# [128,0,0] indicates aeroplane. I hope you could figure it out.
# More details: https://github.com/torrvision/crfasrnn/issues
pallete = [0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128,
            128,64,128,
            0,192,128,
            128,192,128,
            64,64,0,
            192,64,0,
            64,192,0,
            192,192,0]

mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

# Rearrange channels to form BGR
im = image[:,:,::-1]
# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
# Get predictions
segmentation  = net.predict([im])
segmentation2 = segmentation[0:cur_h, 0:cur_w]
output_im     = PILImage.fromarray(segmentation2)
output_im.putpalette(pallete)
out_path      = 'output.png'
output_im.save(out_path)
# plt.imshow(output_im)
# plt.savefig(out_path)