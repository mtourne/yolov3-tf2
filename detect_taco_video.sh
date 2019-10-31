#!/bin/bash

# change class_count (80 or -1 to go between fine_tune and darknet)

python detect_video.py --video 0 \
    --classes /opt/data/TACO/tf_dataset_classes.names \
    --classes_count 80 \
    --weights $1