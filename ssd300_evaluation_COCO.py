import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import warnings
import logging

# TODO: Specify the directory that contains the `pycocotools` here.
pycocotools_dir = '../cocoapi/PythonAPI/'
if pycocotools_dir not in sys.path:
    sys.path.insert(0, pycocotools_dir)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json

# Erase terminal warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)

# %matplotlib inline

# Set the input image size for the model.
img_height = 300
img_width = 300

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=80,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # The scales for Pascal VOC are [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = '/home/gennosuke/ssd-keras-master-v2/weights/VGG_coco_SSD_300x300_iter_400000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# model.summary()

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
MS_COCO_dataset_images_dir = '/home/gennosuke/ROLO-master/benchmark/cocoData/val2017/'
MS_COCO_dataset_annotations_filename = '/home/gennosuke/ROLO-master/benchmark/cocoData/annotations/instances_val2017.json'

dataset.parse_json(images_dirs=[MS_COCO_dataset_images_dir],
                   annotations_filenames=[MS_COCO_dataset_annotations_filename],
                   ground_truth_available=False, # It doesn't matter whether you set this `True` or `False` because the ground truth won't be used anyway, but the parsing goes faster if you don't load the ground truth.
                   include_classes='all',
                   ret=False)

# We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(MS_COCO_dataset_annotations_filename)

# TODO: Set the desired output file name and the batch size.
results_file = 'detections_val2017_ssd300_results.json'
batch_size = 20 # Ideally, choose a batch size that divides the number of images in the dataset.

predict_all_to_json(out_file=results_file,
                    model=model,
                    img_height=img_height,
                    img_width=img_width,
                    classes_to_cats=classes_to_cats,
                    data_generator=dataset,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    model_mode='inference',
                    confidence_thresh=0.01,
                    iou_threshold=0.45,
                    top_k=200,
                    normalize_coords=True)

coco_gt   = COCO(MS_COCO_dataset_annotations_filename)
coco_dt   = coco_gt.loadRes(results_file)
image_ids = sorted(coco_gt.getImgIds())

cocoEval = COCOeval(cocoGt=coco_gt,
                    cocoDt=coco_dt,
                    iouType='bbox')
cocoEval.params.imgIds  = image_ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()