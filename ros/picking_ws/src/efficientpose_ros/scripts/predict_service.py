#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from model import build_EfficientPose
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
import os
import cv2
from functions_prediction import preprocess,postprocess, init_model, build_model,get_linemod_3d_bboxes
import time
from efficientpose_test.msg import DetectionInfo
from efficientpose_test.srv import InputPrediction,InputPredictionResponse
import argparse
from functions_visual import create_visual


class chess_detector:
    def __init__(self, args,class_to_3d_bboxes):

        self.model, self.image_size = build_model(args.weights)

        self.session = tf.compat.v1.keras.backend.get_session()
        self.bridge = CvBridge()
        service_name = "predictions_{}".format(args.object)
        self.service = rospy.Service(service_name, InputPrediction, self.prediction)
        self.class_to_3d_bboxes = class_to_3d_bboxes


    def prediction(self,request):
        print('making predictions')
        score_threshold = 0.5
        cv_image = self.bridge.imgmsg_to_cv2(request.predicted_image, "bgr8")
        print(request.K_matrix)
        K_matrix = np.array([[request.K_matrix[0], 0.,request.K_matrix[2]], [0., request.K_matrix[4], request.K_matrix[5]], [0., 0., 1.]], dtype = np.float32)
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)
            input_list, scale = preprocess(cv_image, self.image_size, K_matrix, 1)
            boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
            boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)           
            image = create_visual(cv_image, rotations, boxes, scores, translations,self.class_to_3d_bboxes, K_matrix)

            response = InputPredictionResponse()
            response.predicted_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            response.depth_map = request.depth_map
            response.rotations = np.array(rotations).flatten()
            response.translations =  np.array(translations).flatten()
            response.scores = scores
            response.boxes = np.array(boxes).flatten()
            response.predicted_image_bbox = self.bridge.cv2_to_imgmsg(image, "bgr8")

        return response


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='arguments for service')
        parser.add_argument('--weights', default="/home/max/Documents/ros_workspaces/zed_ws/src/efficientpose_test/scripts/horse/phi_0_linemod_best_ADD.h5", help='path for weights')
        parser.add_argument('--object', default="horse", help='which object')
        args = parser.parse_args()
        name_node = "detection_node_service_{}".format(args.object)
        print(f"starting node {name_node}")
        rospy.init_node(name_node)
        _ = init_model()

        name_to_3d_bboxes = get_linemod_3d_bboxes()
        class_to_name = {0: args.object}

        class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 

        detector = chess_detector(args,class_to_3d_bboxes)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

