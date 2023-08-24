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

def draw_bbox_8_2D(draw_img, bbox_8_2D, color = (0, 255, 0), thickness = 2):
        """ Draws the 2D projection of a 3D model's cuboid on an image with a given color.

        # Arguments
            draw_img     : The image to draw on.
            bbox_8_2D    : A [8 or 9, 2] matrix containing the 8 corner points (x, y) and maybe also the centerpoint.
            color     : The color of the boxes.
            thickness : The thickness of the lines to draw boxes with.
        """
        #convert bbox to int and tuple
        bbox = np.copy(bbox_8_2D).astype(np.int32)
        bbox = tuple(map(tuple, bbox))
        
        
        #lower level
        cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
        cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
        cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
        cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
        #upper level
        cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
        cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
        cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
        cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
        #sides
        cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
        cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
        cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
        cv2.line(draw_img, bbox[3], bbox[7], color, thickness)
        
        #check if centerpoint is also available to draw
        if len(bbox) == 9:
            #draw centerpoint
            cv2.circle(draw_img, bbox[8], 3, color, -1)

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

        points_bbox_3D = points_bbox_3D/1000
        points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, None)
        points_bbox_2D = np.squeeze(points_bbox_2D)
        
        return points_bbox_2D


def create_visual(image, rotations, boxes, scores, translations, class_to_3d_bboxes, K_matrix):
        if len(boxes) != 0:
            for i in range(len(boxes)):
                if scores[i] > 0.1:
                    x1, y1, x2, y2 = [boxes[i][0],boxes[i][1], boxes[i][2], boxes[i][3]]

                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))

                    # Draw the rectangle on the image
                    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

                    # Draw the 3D axis
                    rotation_vector = np.array(rotations[i])
                    translation_vector = np.array(translations[i]) / 2


                    points_bbox_3D = class_to_3d_bboxes[0]  # Assuming there's only one class (ape)
                    points_bbox_2D = project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, K_matrix, append_centerpoint=True)
                    draw_bbox_8_2D(image, points_bbox_2D)

                    # Draw the axis on the image
                    center_point = tuple(map(int, points_bbox_2D[-1]))  # Last point is the center point

                    axis_length = 100  # Set the desired length of the 3D axis (you can adjust this value as needed)

                    # Extend the end points of the lines
                    x_axis_end = tuple(map(int, points_bbox_2D[0] + axis_length * np.array([1, 0])))  # X-axis is the first point
                    y_axis_end = tuple(map(int, points_bbox_2D[1] + axis_length * np.array([0, 1])))  # Y-axis is the second point
                    z_axis_end = tuple(map(int, points_bbox_2D[2] + axis_length * np.array([0, 0])))  # Z-axis is the third point

                    # X-axis (Red)
                    cv2.line(image, center_point, x_axis_end, (0, 0, 255), 5)
                    # Y-axis (Green)
                    cv2.line(image, center_point, y_axis_end, (0, 255, 0), 5)
                    # Z-axis (Blue)
                    cv2.line(image, center_point, z_axis_end, (255, 0, 0), 5)

        return image

def split_tuple_into_arrays(my_tuple, array_length):
    arrays = [np.array(my_tuple[i:i+array_length]) for i in range(0, len(my_tuple), array_length)]
    return arrays