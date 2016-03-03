#!/bin/bash
# Usage:
# .sh crfasrnn_batch_demo.sh

gpu=0

pts="../origin.models/TVG_CRFRNN_COCO_VOC.prototxt"

model="../origin.models/TVG_CRFRNN_COCO_VOC.caffemodel"

path="/home/ddk/dongdk/dataset/FLIC/crop.images2/test/"

path2="/home/ddk/dongdk/dataset/FLIC/crop.images2/crfasrnn.test/"

python crfasrnn_batch_demo.py --gpu   $gpu \
									            --pts   $pts \
									            --model $model \
									            --path  $path \
									            --path2 $path2