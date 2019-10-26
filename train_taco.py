import sys
import csv
import multiprocessing
import os
import logging

import numpy as np
import tensorflow as tf
import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from imgaug import augmenters as iaa

import yolov3_tf2.models as yolo_model

## TODO turn TACO/detector into a module
TACO_DETECTOR_PATH = '/opt/data/TACO/detector'
sys.path.append(TACO_DETECTOR_PATH)

# TACO IMPORTS
from dataset import Taco
import utils as taco_utils
import model2

IMG_SIZE = 416

class Config(object):
    NAME = "TACO - Yolo experiment 1"

    # eagerly for debug?
    RUN_EAGERLY = True

    # default learning rate from yolov3
    LEARNING_RATE = 1e-3

    EPOCHS = 20
    BATCH_SIZE = 8

    IMAGE_MIN_DIM = IMG_SIZE
    IMAGE_MAX_DIM = IMG_SIZE
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    ## TODO implement "nolding" (whitening) of the data from TACO
    ##      verify the correct average pixel from taco

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

######################
### CONFIGURATION  ###
######################
CONFIG = Config()

def data_generator(dataset, config, shuffle=True, augmentation=None,
                   batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding boxes

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                model2.load_image_gt(dataset, config, image_id,
                              augmentation=augmentation)


            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue


            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.float32)
                ## Don't return the masks to save some memory.

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            
            batch_image_meta[b] = image_meta
            image = model2.mold_image(image.astype(np.float32), config)
            batch_images[b] = image
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids

            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_gt_class_ids, batch_gt_boxes ]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def main():
    dataset = '/opt/data/TACO/data'
    round = 0
    # simplest map
    class_map_file = '/opt/data/TACO/detector/taco_config/map_3.csv'
    use_transplants = False

    augmentation_pipeline = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
                iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
                # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                iaa.Fliplr(0.5),
                iaa.Add((-20, 20),name="Add"),
                iaa.Multiply((0.8, 1.2), name="Multiply"),
                iaa.Affine(scale=(0.8, 2.0)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
            ], random_order=True)

     # Read map of target classes
    class_map = {}
    with open(class_map_file) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}

    # Training dataset.
    dataset_train = Taco()
    dataset_train.load_taco(dataset, round, "train", class_map=class_map, auto_download=None)
    if use_transplants:
        dataset_train.add_transplanted_dataset(use_transplants, class_map=class_map)
    dataset_train.prepare()
    nr_classes = dataset_train.num_classes

    # Validation dataset
    dataset_val = Taco()
    dataset_val.load_taco(dataset, round, "val", class_map=class_map, auto_download=None)
    dataset_val.prepare()

    # Training generator
    image_ids = np.copy(dataset_train.image_ids)

    image_id = 0
    (image,
     image_meta,
     gt_class_ids,
     gt_boxes,
     gt_masks) = model2.load_image_gt(dataset_train, CONFIG, image_id,
                                      augmentation=augmentation_pipeline)

    box = gt_boxes[0]
    print(box)
    # matplotlib : show image
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(image)
    # matlpotlib add bbox and save output
    color = colorsys.hsv_to_rgb(np.random.random(),1,1)
    # (x1,y1) top left corner
    # (x2,y2) bottom right
    [y1, x1, y2, x2] = box
    h = y2 - y1
    w = x2 - x1
    rect = Rectangle((x1,y1+h), w, -h, linewidth=2, edgecolor=color,
                                 facecolor='none', alpha=0.7, linestyle = '--')
    ax.add_patch(rect)
    plt.savefig('dataset_taco_output.png')

    # Data generator
    train_generator = data_generator(dataset_train, CONFIG, shuffle=True,
                                     augmentation=augmentation_pipeline,
                                     batch_size=CONFIG.BATCH_SIZE)
    val_generator = data_generator(dataset_val, CONFIG, shuffle=True,
                                   batch_size=CONFIG.BATCH_SIZE)
    # Callbacks
    callbacks = [
        ## TODO add again
        ## keras.callbacks.TensorBoard(log_dir=self.log_dir,
        ##                          histogram_freq=0, write_graph=True, write_images=False),
        ## keras.callbacks.ModelCheckpoint(self.checkpoint_path,
        ##                              verbose=0, save_weights_only=True),

        # XXX add early stopping?
    ]

    # Work-around for Windows: Keras fails on Windows when using
    # multiprocessing workers. See discussion here:
    # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
    if os.name is 'nt':
        workers = 0
    else:
        workers = multiprocessing.cpu_count()

    # model
    ## NOTE : when doing transfer learning, either only transfer the darkweb
    ## and have custom amount of classes
    ## or keep 80 classes (orig model) and do fine-tuning.
    classes_count = len(class_map)
    model = yolo_model.YoloV3(IMG_SIZE, training=True, classes=classes_count)
    anchors = yolo_model.yolo_anchors
    anchor_masks = yolo_model.yolo_anchor_masks
    
    optimizer = tf.keras.optimizers.Adam(lr=CONFIG.LEARNING_RATE)
    loss = [yolo_model.YoloLoss(anchors[mask], classes=classes_count) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=CONFIG.RUN_EAGERLY)

    epoch=0
    model.fit_generator(
            train_generator,
            initial_epoch=epoch,
            epochs=CONFIG.EPOCHS,
            steps_per_epoch=CONFIG.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=CONFIG.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
    )
    epoch = max(epoch, CONFIG.EPOCHS)


if __name__ == "__main__":
    main()
