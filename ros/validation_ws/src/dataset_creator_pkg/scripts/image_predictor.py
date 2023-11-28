#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
import os
import cv2
from functions_prediction import preprocess,postprocess, init_model, build_model,get_linemod_3d_bboxes

from functions_visual import create_visual
import time
from panda_msgs.msg import DetectionInfo
from std_msgs.msg import Header
import argparse
import pandas as pd
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
import open3d as o3d
from point_cloud_functions import create_pointcloud_msg,sphere_points_around_object, icp_pointclouds
from panda_msgs.srv import InputPrediction,InputPredictionResponse

class chess_detector:
    def __init__(self, args):
        self.name = args.object
        self.bridge = CvBridge()
        self.phi = 0
        self.score_threshold = 0.5
        self.K_matrix = None
        self.depth_data = None

        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        
        self.pub_depth_map = rospy.Publisher('/Pointcloud_Publisher_depth_map', PointCloud2,queue_size=10)
        self.depth_map_points = None
        print('predict')

        while not rospy.is_shutdown():
            self.run()  

    def depth_callback(self,data):
        self.depth_data = data
    
    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                fx=data.K[0], fy=data.K[4],
                                                cx=data.K[2], cy=data.K[5])
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            print("The k matrix: ", self.K_matrix)

    def create_pc_from_depth_image(self, data):
        # Get a pointer to the depth values casting the data pointer to floating point
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(np.asarray(data)),self.camera_intrinsic)
        depth_map_points = np.asarray(pcd.points)

        return depth_map_points 

    def run(self):
        depth_image_filename = "/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/depth/image_0000.png"
        depth_image = cv2.imread(depth_image_filename, cv2.IMREAD_ANYDEPTH)
        o3d.geometry.Image(np.asarray(data))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(np.asarray(data)),self.camera_intrinsic)

        print(depth_image)
        image_filename = "/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/raw_images/image_0000.png"
        image = cv2.imread(image_filename)
        image = self.bridge.cv2_to_imgmsg(image, "bgr8")
    
        self.depth_map_points = self.create_pc_from_depth_image(depth_image)
        pc_msg_depth = create_pointcloud_msg('zed2_left_camera_optical_frame',self.depth_map_points)
        self.pub_depth_map.publish(pc_msg_depth)

        response = self.prediction(image, pc_msg_depth)
        print('predict')

    def handle_6D_prediction_request(self, image, pc_msg_depth):
        prediction_service = rospy.ServiceProxy("predictions_horse", InputPrediction)
        # Create a request object
        request = InputPrediction._request_class()
        request.predicted_image = image
        request.depth_map = pc_msg_depth
        request.K_matrix = self.K_matrix.flatten()
        # Call the service with the request
        response = prediction_service(request)
        self.pose_publisher.publish(response.predicted_image_bbox)
        return response

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='arguments for service')
        parser.add_argument('--weights', default="/home/max/Documents/ros_workspaces/zed_ws/src/efficientpose_ros/scripts/horse/phi_0_linemod_best_ADD.h5", help='path for weights')
        parser.add_argument('--object', default="horse", help='path for weights')
        
        args = parser.parse_args()
        rospy.init_node("detection_node",anonymous = True)
        class_to_name = {0: args.object}

        detector = chess_detector(args)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

