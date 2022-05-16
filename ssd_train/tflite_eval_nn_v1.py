### Copyright (C) 2020 GreenWaves Technologies
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##############################################################################

# 1. General requirements 
import os
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from models.ssd_model import build_models
from models.ssd_model_utils import *
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
import tensorflow as tf
import cv2

##############################################################################
# 2. Set model configuration parameters 
img_height = 80 # Height of the input images
img_width = 80 # Width of the input images
img_channels = 1 # Number of color channels of the input images
intensity_mean = 0 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 256 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1 # Number of positive classes
scales = [0.05, 0.15, 0.3, 0.4, 0.6] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
batch_size = 8
# The list of aspect ratios for the anchor boxes
aspect_ratios_global = None
aspect_ratios_per_layer = [[1./4., 1./3., 1./2., 1., 2., 3., 4.],
                           [1./4., 1./3., 1./2., 1., 2., 3., 4.],
                           [1./4., 1./3., 1./2., 1., 2., 3., 4.],
                           [1./4., 1./3., 1./2., 1., 2., 3., 4.]]
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [.1, .1, .1, .1] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
n_predictor_layers = 4
training_info_path = './main_model_training_info'
if not os.path.isdir(training_info_path):
    os.mkdir(training_info_path)


##############################################################################
# 3. Models instantiation
    
# pass model configuration parameters to the model builder

# step1: clear previous models from memory.
K.clear_session() 

image_size=(img_height, img_width, img_channels)
model_constructor_params = {'image_size': image_size, 'n_classes': n_classes, 'mode':'training', 'training_info_path':training_info_path, \
                            'l2_regularization':0.0005, 'min_scale': 0.1, 'max_scale':0.9, 'scales':scales, 'n_predictor_layers':n_predictor_layers,\
                            'aspect_ratios_global':aspect_ratios_global,'aspect_ratios_per_layer':aspect_ratios_per_layer, 'two_boxes_for_ar1':True,\
                            'steps':None, 'offsets':None, 'clip_boxes':False, 'variances':variances, 'coords':'centroids',\
                            'normalize_coords':True, 'subtract_mean':intensity_mean,'divide_by_stddev':intensity_range,'swap_channels':False,\
                            'confidence_thresh':0.5,'iou_threshold':0.3, 'top_k':40, 'nms_max_output_size':400,\
                            'return_predictor_sizes':True,'build_base_model':True}

# step 2: Pass model configuration parameters to the model builder
constructed_models = build_models(model_constructor_params)
main_model =constructed_models.main_model
predictor_sizes = constructed_models.predictor_sizes

##############################################################################

def preprocess(resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return  resized_inputs/128 - 1.0

def construct_input_tensor(img_path, input_details):
    'load an image and rehsape according to the model input layer'
    img_in = cv2.imread(img_path,0)


    img_in = cv2.resize(img_in,(80,80))
    img = preprocess(img_in)
    input_array = np.reshape(img, input_details[0]['shape'])
    image = input_array.astype(np.float32)
    return image

#%%############################################################################ 
# Evaluation


img_path = 'data/images/image_room1_0_254.png'

def run_nn_v1(img_path, model_file):
    K.clear_session() 
    interpreter = tf.lite.Interpreter(model_path=model_file)
    all_tensor_details = interpreter.get_tensor_details()
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    input_array = construct_input_tensor(img_path, input_details)
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    classes_out_1 = interpreter.get_tensor(output_details[0]['index'])
    classes_out_2 = interpreter.get_tensor(output_details[1]['index'])
    classes_out_3 = interpreter.get_tensor(output_details[2]['index'])
    classes_out_4 = interpreter.get_tensor(output_details[3]['index'])
    boxes_out_1 = interpreter.get_tensor(output_details[4]['index'])
    boxes_out_2 = interpreter.get_tensor(output_details[5]['index'])
    boxes_out_3 = interpreter.get_tensor(output_details[6]['index'])
    boxes_out_4 = interpreter.get_tensor(output_details[7]['index'])
    classes_tflite = [classes_out_1,
        classes_out_2,
        classes_out_3,
        classes_out_4,
        ]
    bboxes_tflite = [boxes_out_1,
        boxes_out_2,
        boxes_out_3,
        boxes_out_4,
        ]

    input_layer_constructor = construct_input_layer(model_constructor_params)
    feature_maps_constructor = construct_feature_maps(model_constructor_params)
    predictors_constructor = construct_predictors(model_constructor_params)
    anchors_constructor = construct_default_anchors(model_constructor_params)
    output_constructor = construct_model_output(model_constructor_params)

    classes2 = [
        Input(shape=(40, 40, 16)),
        Input(shape=(20, 20, 16)),
        Input(shape=(10, 10, 16)),
        Input(shape=(5, 5, 16)),
    ]
    bboxes2 = [
        Input(shape=(40, 40, 32)),
        Input(shape=(20, 20, 32)),
        Input(shape=(10, 10, 32)),
        Input(shape=(5, 5, 32)),
    ]

    # model construction
    anchors2 = anchors_constructor(bboxes2)
    predictions2      = output_constructor(classes2, bboxes2, anchors2)

    main_model2 = Model(inputs=classes2 + bboxes2, outputs=predictions2)

    session = tf.Session()
    keras.backend.set_session(session)
    with session.as_default():
        with session.graph.as_default():
            main_model2 = Model(inputs=classes2 + bboxes2, outputs=predictions2)
            y_pred = main_model2.predict(classes_tflite + bboxes_tflite)
            y_pred_decoded = decode_detections(y_pred,
                               confidence_thresh=0.5,
                               iou_threshold=0.3,
                               top_k=40,
                               normalize_coords=normalize_coords,
                               img_height=img_height,
                               img_width=img_width)
            bbox = []
            for row in y_pred_decoded[0]:
                score = row[1]
                x_min = int(row[2])
                y_min = int(row[3])
                x_max = int(row[4])
                y_max = int(row[5])
                new_row = [x_min, y_min, x_max, y_max, 'person', score]
                bbox.append(new_row)
                if False:
                    cv2.rectangle(gray, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                    cv2.imshow('test', gray)
                    cv2.waitKey(0)
            return bbox
    return None

if __name__ == '__main__':
    run_nn_v1(img_path)
