#!/bin/bash
mkdir -p ./data 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 # Explore the dataset

python ../../../yolov3-tf2/tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python ../../../yolov3-tf2/tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord

wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights # get pretrained weights

python ../../../yolov3-tf2/convert.py # default save to ./checkpoints/yolov3.tf

python transfer_learning_train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 2 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80
