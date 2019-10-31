#!/bin/bash

# use fine_tune - requires to have as many classes as existing yolo (80 classes)

python train.py \
    --dataset /opt/data/TACO/tf_dataset_train.bin \
    --val_dataset /opt/data/TACO/tf_dataset_val.bin \
    --weights ./checkpoints/yolov3.tf \
    --classes /opt/data/TACO/tf_dataset_classes.names \
    --classes_count 80 \
    --mode fit \
    --transfer fine_tune \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-3