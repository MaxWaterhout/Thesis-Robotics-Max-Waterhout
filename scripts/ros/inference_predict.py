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
from functions_prediction import preprocess,postprocess, init_model, build_model
import time
from efficientpose_test.msg import DetectionInfo
from std_msgs.msg import Header


class chess_detector:
    def __init__(self,num_classes,class_to_3d_bboxes):

        self.model, self.image_size = build_model(num_classes)

        self.session = tf.compat.v1.keras.backend.get_session()

        self.bridge = CvBridge()
        self.predict_pub = rospy.Publisher("image_predictions", DetectionInfo)
        self.phi = 0
        self.score_threshold = 0.5
        self.K_matrix = None
        self.depth_data = None

        self.class_to_3d_bboxes = class_to_3d_bboxes
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
    
    def depth_callback(self,data):
        self.depth_data = data
    
    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            print("The k matrix: ", self.K_matrix)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        msg = self.prediction(cv_image)
        self.predict_pub.publish(msg)


    def prediction(self,cv_image):
        msg = DetectionInfo()
        score_threshold = 0.5
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)
            input_list, scale = preprocess(cv_image, self.image_size, self.K_matrix, 1)
            boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
            boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)
            
            msg.header = Header(stamp=rospy.Time.now(), frame_id="your_frame_id")
            msg.predicted_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            msg.rotations = np.array(rotations).flatten()
            msg.translations = np.array(translations).flatten()

            msg.boxes = np.array(boxes).flatten()
            msg.scores = scores
            msg.depth_map = self.depth_data

        return msg

if __name__ == '__main__':
    try:
        rospy.init_node("detection_node",anonymous = True)
        num_classes, class_to_3d_bboxes = init_model()
        detector = chess_detector(num_classes, class_to_3d_bboxes)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

