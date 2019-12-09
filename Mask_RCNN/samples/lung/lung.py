"""
Mask R-CNN
Configurations and data loading.
Licensed under the MIT License (see LICENSE for details)
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python lung.py train --dataset=/path/to/image/folder --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python lung.py train --dataset=/path/to/image/folder --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python lung.py train --dataset=/path/to/image/folder --model=/path/to/weights.h5

    # Run evaluatoin on the last model you trained
    python lung.py evaluate --dataset=/path/to/image/folder --model=last
"""

import os
import sys
import time
import math
import numpy as np
import skimage.io
import imgaug  # https://github.com/aleju/imgaug (pip install imgaug)
import pandas as pd
import zipfile
import urllib.request
import shutil
import datetime
import matplotlib.pyplot as plt
from time import time
# from keras.callbacks import LearningRateScheduler


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "results/lungs/")
RESULTS_DIR = os.path.join(ROOT_DIR, "results/lung_vis/")
############################################################
#  Configurations
############################################################
from imgaug import augmenters as iaa

class LungConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "lung"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

    STEPS_PER_EPOCH = 111

    DETECTION_MIN_CONFIDENCE = 0

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1

############################################################
#  Dataset
############################################################

class LungDataset(utils.Dataset):

    def load_lung(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        """

        self.add_class("lung", 1, "lung")
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Get image ids from directory names
        image_ids = next(os.walk(dataset_dir))[2]

        for image_id in image_ids:
            self.add_image(
                "lung",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id))


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        # Read mask files from .png image
        mask = skimage.io.imread(os.path.join(mask_dir, info['id'])).astype(np.bool)
        mask = np.expand_dims(mask, axis=2)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lung":
            return info["id"]
        else:
            super().image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LungDataset()
    dataset_train.load_lung(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LungDataset()
    dataset_val.load_lung(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network all")
    model.train(dataset_train, dataset_val,
                learning_rate=Config.LEARNING_RATE,
                epochs=30,
                layers="4+", custom_callbacks= None, augmentation = imgaug.augmenters.Fliplr(0.5))

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None):
    assert image_path
    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    
    print("Saved to ", file_name)


def detect(model, visualization = True):
    """Run detection on images in the given directory."""
    
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    Read dataset
    dataset = LungDataset()
    dataset.load_lung(args.dataset, "test")
    dataset.prepare()

    overlaps = []
    dif = []
    pred_prob = []
    imageID = []
    dices = []
    for image_id in dataset.image_ids:
        image, _, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_differences(image,
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            dataset.class_names, title="", ax=None,
            show_mask=True, show_box=True,
            iou_threshold=0.5, score_threshold=0.5)
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        iou = utils.compute_overlaps_masks(gt_mask, r['masks'])
        dices.append(visualize.Dice(gt_mask, r['masks']))
        # print(iou)
        if not iou:
            print(image_id, dataset.image_info[image_id])
            overlaps.extend([0])
            break
        overlaps.extend(iou)
        dif1 = np.subtract(gt_mask.flatten(), r['masks'].flatten(), dtype=np.float)
        dif.append(np.sum(dif1))
        pred_prob.extend(r['scores'])
        imageID.append(dataset.image_info[image_id]["id"])
    try:
        overlaps = np.squeeze(np.asarray(overlaps), axis=1)
    except:
        overlaps = np.asarray(overlaps) 
    dif = np.asarray(dif)
    dices = np.asarray(dices)
    pred_prob = np.asarray(pred_prob)
    imageID = np.asarray(imageID)
    print('Mean IoU:', np.mean(overlaps))
    print('Mean Dice:', np.mean(dices))

    df = pd.DataFrame({"imageID": imageID, "ious" : overlaps, "prob" : np.asarray(pred_prob),"dif(gt-pred)": dif})
    des_dir = os.path.join(ROOT_DIR, "results/lung_pred_csv/{:%Y%m%dT%H%M%S}.csv".format(datetime.datetime.now()))
    df.to_csv(des_dir,index=False)
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LungConfig()
    else:
        class InferenceConfig(LungConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    elif args.command == "test":
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = "COCO_WEIGHTS"
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        detect(model)
    else:
        print("'{}' is not recognized. "
                "Use 'train' or 'test'".format(args.command))



