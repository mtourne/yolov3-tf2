#!/bin/bash

python train.py \
    --dataset /opt/data/TACO/tf_dataset_train.bin \
    --val_dataset /opt/data/TACO/tf_dataset_val.bin \
    --weights ./checkpoints/yolov3.tf \
    --classes /opt/data/TACO/tf_dataset_classes.names \
    --classes_count -1 \
    --mode fit \
    --transfer darknet \
    --epochs 2000 \
    --batch_size 10 \
    --learning_rate 1e-4