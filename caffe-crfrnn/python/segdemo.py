caffe_root = '/home/sadeep/Desktop/crf-rnn-web-demo/caffe-fcn-sadeep/'
import sys
sys.path.insert(0,caffe_root+'python')

import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import exifutil

import caffe


  
MODEL_FILE = '/home/sadeep/Desktop/crf-rnn-web-demo/caffe-fcn-sadeep/models/crf_rnn/fcn-8s-pascal-deploy.prototxt'
PRETRAINED = '/home/sadeep/Desktop/crf-rnn-web-demo/caffe-fcn-sadeep/models/crf_rnn/fcn-8s-pascal.caffemodel'
IMAGE_FILE = '/home/sadeep/Desktop/crf-rnn-web-demo/caffe-fcn-sadeep/models/crf_rnn/2007_000033.jpg'

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
           
net = caffe.Segmenter(MODEL_FILE, PRETRAINED, gpu=False)

input_image = 255 * exifutil.open_oriented_im(IMAGE_FILE)    


# Mean values in BGR format
mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1,1,3);

# Rearrange channels to form BGR
im = input_image[:,:,::-1]

# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

# Get predictions
segmentation = net.predict([im])

output_im = Image.fromarray(segmentation)

output_im.putpalette(pallete);

output_im.save('hahasadeep.png')