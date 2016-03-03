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
import argparse
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import matplotlib.pyplot as plt
import caffe
from time import sleep


# Each row represents one class. For example, [0, 0, 0] means background. 
# [128,0,0] indicates aeroplane. I hope you could figure it out.
# More details: https://github.com/torrvision/crfasrnn/issues
# 
# Pixel indices correspond to classes in alphabetical order
# More details: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
#   1=aeroplane,     2=bicycle, 3=bird,   4=boat,       5=bottle, 
#   6=bus,           7=car ,    8=cat,    9=chair,     10=cow, 
#  11=diningtable,  12=dog,    13=horse, 14=motorbike, 15=person, 
#  16=potted plant, 17=sheep,  18=sofa,  19=train,     20=tv/monitor

# ## origin
# pallete = [0,   0,   0,   #  0 - bg
#            128, 0,   0,   #  1 - aeroplane
#            0,   128, 0,   #  2 - bicycle
#            128, 128, 0,   #  3 - bird
#            0,   0,   128, #  4 - boat
#            128, 0,   128, #  5 - bottle
#            0,   128, 128, #  6 - bus
#            128, 128, 128, #  7 - car
#            64,  0,   0,   #  8 - cat
#            192, 0,   0,   #  9 - chair
#            64,  128, 0,   # 10 - cow
#            192, 128, 0,   # 11 - diningtable
#            64,  0,   128, # 12 - dog
#            192, 0,   128, # 13 - horse
#            64,  128, 128, # 14 - motorbike
#            192, 128, 128, # 15 - person
#            0,   64,  0,   # 16 - potted plant
#            128, 64,  0,   # 17 - sheep
#            0,   192, 0,   # 18 - sofa
#            128, 192, 0,   # 19 - train
#            0,   64,  128, # 20 - tv/monitor
#            128, 64,  128, # ...
#            0,   192, 128,
#            128, 192, 128,
#            64,  64,  0,
#            192, 64,  0,
#            64,  192, 0,
#            192, 192, 0]

# ## only person cls
pallete = [0,   0,   0,   #  0 - bg
           0,   0,   0,   #  1 - aeroplane
           0,   0,   0,   #  2 - bicycle
           0,   0,   0,   #  3 - bird
           0,   0,   0,   #  4 - boat
           0,   0,   0,   #  5 - bottle
           0,   0,   0,   #  6 - bus
           0,   0,   0,   #  7 - car
           0,   0,   0,   #  8 - cat
           0,   0,   0,   #  9 - chair
           0,   0,   0,   # 10 - cow
           0,   0,   0,   # 11 - diningtable
           0,   0,   0,   # 12 - dog
           0,   0,   0,   # 13 - horse
           0,   0,   0,   # 14 - motorbike
           255, 255, 255, # 15 - person
           0,   0,   0,   # 16 - potted plant
           0,   0,   0,   # 17 - sheep
           0,   0,   0,   # 18 - sofa
           0,   0,   0,   # 19 - train
           0,   0,   0,   # 20 - tv/monitor
           0,   0,   0,   # ...
           0,   0,   0,  
           0,   0,   0,  
           0,   0,   0,
           0,   0,   0,
           0,   0,   0,
           0,   0,   0]


def mkdir(path):
  if not os.path.isdir(path):
    os.makedirs(path)  
  else:
    print path, "has exists."

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--gpu',   dest='gpu_id', help='GPU device id to use [0]',
                      default=0, type=int)
  parser.add_argument('--cpu',   dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--pts',   dest='pts',   help='Deploy prototxt to use',
                      type=str,  default='')
  parser.add_argument('--model', dest='model', help='Trained model to use',
                      type=str,  default='')
  parser.add_argument('--path',  dest='path',  help='Path to an image or images directory',
                      type=str,  default='')
  parser.add_argument('--path2', dest='path2', help='Path2 to save the output of input image(s) by the net',
                      type=str,  default='')
  args = parser.parse_args()

  pts   = args.pts.strip()
  if not os.path.isfile(pts):
      raise IOError(('{:s} not found.').format(pts))

  model = args.model.strip()
  if not os.path.isfile(model):
      raise IOError(('{:s} not found.').format(model))

  path  = args.path.strip()
  if os.path.isfile(path):
    im_paths = [path]
  elif os.path.isdir(path):
    im_paths = [path + file.strip() for file in os.listdir(path)]
  else:
    raise IOError(('{:s} not exist').format(path))
  im_paths.sort()

  path2 = args.path2.strip()
  mkdir(path2)

  if args.cpu_mode:
    gpu = False
    print "use cpu mode"
  else:
    gpu = True
    print "use gpu mode"
  sleep(3)
  net = caffe.Segmenter(pts, model, gpu, MaxDim=500)
  print '\n\nLoaded deploy prototxt {:s} done.'.format(pts)
  print '\n\nLoaded network {:s} done.'.format(model)
  
  return args, net, im_paths, path2

def crfasrnn_batch_demo(s_im_ext=".png"):
  ''''''
  args, net, im_paths, s_path = parse_args()

  c = 0
  mean_vec          = np.array([103.939, 116.779, 123.68], dtype=np.float32)
  reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

  im_num = len(im_paths)
  for im_path in im_paths:
    print "im: %s (%s) path: %s" % (c, im_num, im_path)
    c +=1 

    in_im    = 255 * caffe.io.load_image(im_path)
    image = PILImage.fromarray(np.uint8(in_im))
    image = np.array(image)

    # Rearrange channels to form BGR
    im = image[:,:,::-1]
    
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    m_side              = max(cur_h, cur_w)
    pad_h = m_side - cur_h
    pad_w = m_side - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
   
    # Get predictions
    print "predict - MaxDim:", m_side
    segmentation  = net.predict([im], m_side)
    segmentation  = segmentation[0:cur_h, 0:cur_w]
    out_im        = PILImage.fromarray(segmentation)
    out_im.putpalette(pallete)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # out_im = out_im.convert('L')
    
    # Save final output
    out_name = os.path.basename(im_path)
    out_path = s_path + out_name.rsplit(".", 1)[0] + s_im_ext
    out_im.save(out_path)
    
    # plt.imshow(out_im)
    # plt.savefig(out_path)
    # plt.clf()
    print "done."

if __name__ == '__main__':
  ''''''
  crfasrnn_batch_demo()