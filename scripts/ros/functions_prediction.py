#!/home/max/env38/bin/python

import cv2
import numpy as np
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from model import build_EfficientPose
from utils import preprocess_image
import time

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)


def get_linemod_3d_bboxes():
    """
    Returns: name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values

    """
    name_to_model_info = {"ape":            {"diameter": 53.408474715846296, "min_x": -12.62461, "min_y": -12.18997, "min_z": -2.835575, "size_x": 24.785775, "size_y": 26.188718, "size_z": 51.373154}}
        
    name_to_3d_bboxes = {name: convert_bbox_3d(model_info) for name, model_info in name_to_model_info.items()}
    
    return name_to_3d_bboxes


def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    #get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]
    
    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox

def preprocess(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for EfficientPose
    Args:
        image: The image to predict
        image_size: Input resolution for EfficientPose
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_list: List containing the preprocessed inputs for EfficientPose
        scale: The scale factor of the resized input image and the original image

    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale


def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]

    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector

def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, append_centerpoint = True):
    """ Projects the 3D model's cuboid onto a 2D image plane with the given rotation, translation and camera matrix.

    Arguments:
        points_bbox_3D: numpy array with shape (8, 3) containing the 8 (x, y, z) corner points of the object's 3D model cuboid 
        rotation_vector: numpy array containing the rotation vector with shape (3,)
        translation_vector: numpy array containing the translation vector with shape (3,)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        append_centerpoint: Boolean indicating wheter to append the centerpoint or not
    Returns:
        points_bbox_2D: numpy array with shape (8 or 9, 2) with the 2D projections of the object's 3D cuboid
    """
    if append_centerpoint:
        points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape = (1, 3))], axis = 0)
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, None)
    points_bbox_2D = np.squeeze(points_bbox_2D)
    
    return points_bbox_2D

def postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold):
    """
    Filter out detections with low confidence scores and rescale the outputs of EfficientPose
    Args:
        boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
        scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
        labels: numpy array [batch_size = 1, max_detections] containing class label
        rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
        scale: The scale factor of the resized input image and the original image
        score_threshold: Minimum score threshold at which a prediction is not filtered out
    Returns:
        boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
        scores: numpy array [num_valid_detections] containing the confidence scores
        labels: numpy array [num_valid_detections] containing class label
        rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [num_valid_detections, 3] containing the translation vectors

    """
    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    # correct boxes for image scale
    boxes /= scale
    #rescale rotations
    rotations *= math.pi
    #filter out detections with low scores
    indices = np.where(scores[:] > score_threshold)
    # select detections
    scores = scores[indices]
    boxes = boxes[indices]
    rotations = rotations[indices]
    translations = translations[indices]
    labels = labels[indices]
    
    return boxes, scores, labels, rotations, translations


def init_model():
    """
    Run EfficientPose in inference mode live on webcam.

    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()
    
    # save_path = "./predictions/occlusion/" #where to save the images or None if the images should be displayed and not saved
    save_path = None
    image_extension = ".jpg"
    class_to_name = {0: "ape"} #Occlusion
    #class_to_name = {0: "driller"} #Linemod use a single class with a name of the Linemod objects
    translation_scale_norm = 1.0
    draw_bbox_2d = True
    draw_name = False
    num_classes = len(class_to_name)

    #you probably need to replace the linemod camera matrix with the one of your webcam
    name_to_3d_bboxes = get_linemod_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    return num_classes, class_to_3d_bboxes 



def build_model(num_classes):
    print("\nBuilding model...\n")
    _, model, _ = build_EfficientPose(0,
                                num_classes = num_classes,
                                num_anchors = 9,
                                freeze_bn = True,
                                score_threshold = 0.5,
                                num_rotation_parameters = 3,
                                print_architecture = False)
    print("\n Build model!")
    print("\n loading weights")
    model.load_weights("/home/max/Documents/ros_workspaces/zed_ws/src/efficientpose_test/scripts/horse/phi_0_linemod_best_ADD.h5", by_name=True)
    print("Done!")
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[0]

    return model, image_size

