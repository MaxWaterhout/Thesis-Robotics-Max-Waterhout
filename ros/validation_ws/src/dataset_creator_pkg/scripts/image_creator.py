#!/home/max/env38/bin/python

import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import os
import cv2
import pandas as pd
from std_msgs.msg import Empty, Int32
import tf
from geometry_msgs.msg import PoseStamped
from point_cloud_functions import create_pointcloud_msg
from sensor_msgs.msg import PointCloud2, CameraInfo
import open3d as o3d
import atexit
import argparse

class image_creator:
    def __init__(self,args):        
        rospy.init_node("dataset_creator")
        print("starting node")
        self.a1 = args.a1
        self.a2 = args.a2
        self.a3 = args.a3
        self.b1 = args.b1
        self.b2 = args.b2
        self.b3 = args.b3

        self.image_name = None
        self.bridge = CvBridge()
        try:
            final_df_filename = "/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/dataset/final_df.csv"
            self.final_df = pd.read_csv(final_df_filename)
        except:
            self.final_df = pd.DataFrame()
        self.listener = tf.TransformListener()
        directory_path = '/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/raw_images/'

        # Use os.listdir to get a list of files in the directory
        files = os.listdir(directory_path)
        # Use a list comprehension to count the number of image files (e.g., .jpg or .png)
        image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        self.image_number = len(image_files)
        print(self.image_number,'Number of images')
        self.activation = rospy.Subscriber('/activation_topic', Int32, self.activation_callback)
        self.camera_intrinsic = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )

        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        self.image_data = None
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.depth_data = None
        self.aruco_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color_charuco_pose", PoseStamped, self.pose_callback)
        self.aruco_pose = None
        self.aruco_image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color_charuco_detection", Image, self.aruco_image_callback)
        self.aruco_pose_image = None
        self.pub_depth_map = rospy.Publisher('/Pointcloud_Publisher_depth_map', PointCloud2,queue_size=10)
        self.depth_map_points = None
        self.br = tf.TransformBroadcaster()
        atexit.register(self.save_final_df)  # Register the save_final_df method to be called on exit

    def camera_callback(self,data):
        if self.camera_intrinsic is None:
            K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            camera_frame_id = data.header.frame_id
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                fx=data.K[0], fy=data.K[4],
                                                cx=data.K[2], cy=data.K[5])
            
            self.fx = data.K[0]
            self.fy = data.K[4]
            self.cx = data.K[2]
            self.cy = data.K[5]

            print("The k matrix: ", K_matrix)
                

    def image_callback(self, data):
        self.image_data = data

    def depth_callback(self,data):
        self.depth_data = data

    def pose_callback(self,data):
        self.aruco_pose = data

    def aruco_image_callback(self,data):
        self.aruco_pose_image = data

    def activation_callback(self, msg):
        rospy.loginfo(f"index {msg.data} of robot coordinate system")
        self.image_name = "image_{:04d}".format(self.image_number)
        # Your activation logic here
        self.save_results(self.depth_data,self.aruco_pose,self.image_data,self.aruco_pose_image)   
        
    def create_pc_from_depth_image(self, data):
        cv_image_depth_raw = self.bridge.imgmsg_to_cv2(data, "32FC1")

        # Get a pointer to the depth values casting the data pointer to floating point
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(np.asarray(cv_image_depth_raw)),self.camera_intrinsic)
        # Define the filename to save the PCD file
        pcd_filename = f"/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/pointcloud/{self.image_name}.pcd"

        # Save the PointCloud to a PCD file
        o3d.io.write_point_cloud(pcd_filename, pcd)

        depth_map_points = np.asarray(pcd.points)

        return depth_map_points

    def red_pixels_exist(self,image):
        # Load the image

        # Define the lower and upper bounds for red color detection (in BGR format)
        lower_red = np.array([0, 0, 200])  # Adjust these values based on your specific red color
        upper_red = np.array([50, 50, 255])  # Adjust these values based on your specific red color

        # Create a mask for red color detection
        red_mask = cv2.inRange(image, lower_red, upper_red)

        # Check if there are any red pixels in the image
        red_pixels_exist = cv2.countNonZero(red_mask) > 0

        if red_pixels_exist:
            return True
        else:
            return False
        

    def save_results(self, depth_data,aruco_pose,image_data,aruco_pose_image):
       
        cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        cv_image_aruco = self.bridge.imgmsg_to_cv2(aruco_pose_image, "bgr8")
        do_red_pixels_exist = self.red_pixels_exist(cv_image_aruco)

        if do_red_pixels_exist:
            self.depth_map_points = self.create_pc_from_depth_image(depth_data)
            pc_msg_depth = create_pointcloud_msg('zed2_left_camera_optical_frame',self.depth_map_points)
            self.pub_depth_map.publish(pc_msg_depth)

            # Save the color image
            color_image_filename = f"/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/raw_images/{self.image_name}.png"
            cv2.imwrite(color_image_filename, cv_image)

            # Save the ArUco pose image
            aruco_image_filename = f"/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/aruco_images/{self.image_name}.png"
            cv2.imwrite(aruco_image_filename, cv_image_aruco)

            self.results_to_df(aruco_pose,do_red_pixels_exist)
            rospy.loginfo(f"Saved {self.image_name}!")
            self.image_number += 1

        else: 
            rospy.loginfo("there is no red in this image")

    def results_to_df(self,aruco_pose,do_red_pixels_exist):
        data_dict = {
        'image_name': [self.image_name],  # Convert to a list
        'Orientation_x': [aruco_pose.pose.orientation.x],  # Convert to a list
        'Orientation_y': [aruco_pose.pose.orientation.y],  # Convert to a list
        'Orientation_z': [aruco_pose.pose.orientation.z],  # Convert to a list
        'Orientation_w': [aruco_pose.pose.orientation.w],  # Convert to a list
        'Trans_x': [aruco_pose.pose.position.x],  # Convert to a list
        'Trans_y': [aruco_pose.pose.position.y],  # Convert to a list
        'Trans_z': [aruco_pose.pose.position.z],  # Convert to a list
        'A1': [self.a1],  # Convert to a list
        'A2': [self.a2],  # Convert to a list
        'A3': [self.a3],  # Convert to a list
        'B1': [self.b1],  # Convert to a list
        'B2': [self.b2],  # Convert to a list
        'B3': [self.b3], # Convert to a list
        'fx':[self.fx],
        'fy':[self.fy],
        'cx':[self.cx],
        'cy':[self.cy],
        'width':[1280],
        'heigth':[720],
        'aruco_pose': [do_red_pixels_exist]
            }
        
        df = pd.DataFrame(data_dict)
        print(df)
        df = df.iloc[[0]]

        self.final_df = pd.concat([self.final_df, df], ignore_index=True)
        self.final_df.to_clipboard(index=False, header=None)

    def save_final_df(self):
        # Save the final_df to a CSV file when the script is about to exit
        final_df_filename = "/home/max/Documents/ros_workspaces/aruco_marker_ws/src/dataset_creator_pkg/dataset/final_df.csv"
        self.final_df.to_csv(final_df_filename, index=False)
        print(f"Saved {len(self.final_df)} rows to {final_df_filename}")
             

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='arguments for service')
        parser.add_argument('--a1', default="fill", help='which object on a1')
        parser.add_argument('--a2', default="fill", help='which object on a2')
        parser.add_argument('--a3', default="fill", help='which object on a3')
        parser.add_argument('--b1', default="fill", help='which object on b1')
        parser.add_argument('--b2', default="fill", help='which object on b2')
        parser.add_argument('--b3', default="fill", help='which object on b3')

        args = parser.parse_args()
        detector = image_creator(args)

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

