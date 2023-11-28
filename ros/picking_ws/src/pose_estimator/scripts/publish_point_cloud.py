#!/home/max/env38/bin/python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import trimesh
from efficientpose_test.msg import DetectionInfo
import cv2
import open3d as o3d
import time
from cv_bridge import CvBridge
import tf
from scipy.spatial.transform import Rotation as Rot
from point_cloud_functions import split_tuple_into_arrays, create_pointcloud_msg, icp_pointclouds,sphere_points_around_object
import argparse


class PointCloudModifier:
    def __init__(self,args):
        rospy.init_node('pointcloud_modifier', anonymous=True)
        self.args = args
        self.pub = rospy.Publisher('/modified_pointcloud_topic', PointCloud2, queue_size=10)
        print("Running with depth: ", self.args.depth)
        print("Running with ICP: ", self.args.icp)
        
        if self.args.depth == True:
            self.pub_depth_map = rospy.Publisher('/Pointcloud_Publisher_depth_map', PointCloud2,queue_size=10)
            self.depth_map_points = None
            self.pub_depth_map_icp = rospy.Publisher('/Pointcloud_Publisher_depth_map_icp', PointCloud2,queue_size=10)
            self.filtered_points = None
            
            if self.args.icp == True:
                self.pub_icp = rospy.Publisher('/modified_pointcloud_topic_icp', PointCloud2, queue_size=1)

                self.new_points_icp = None

        self.camera_intrinsic = None
        self.K_matrix = None
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.bridge = CvBridge()
        
        self.t = None
        self.transformation_matrix = None

        self.new_points = None


        # Load your new points
        self.mesh = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000005.ply')
        self.object_points = np.array(self.mesh.vertices, dtype=np.float32) / 1000
        # Transform the new points to the camera frame
        self.predictions = rospy.Subscriber("image_predictions",DetectionInfo,self.prediction_callback )
        
    def camera_callback(self, data):
                if self.K_matrix is None:
                    self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
                    self.camera_frame_id = data.header.frame_id
                    self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                        fx=data.K[0], fy=data.K[4],
                                                        cx=data.K[2], cy=data.K[5])
                    print("The k matrix: ", self.K_matrix)
                    print("\n The camera frame id:" , self.camera_frame_id)

    def prediction_callback(self,data):
        rotations = split_tuple_into_arrays(data.rotations,3)
        translations = split_tuple_into_arrays(data.translations,3)

        #publish the depth map to a pointcloud
        if self.args.depth == True:
            self.depth_map_points = self.create_pc_from_depth_image(data.depth_map)
            pc_msg_depth = create_pointcloud_msg('zed2_left_camera_frame',self.depth_map_points)
            self.pub_depth_map.publish(pc_msg_depth)

        if len(rotations) != 0:
            # publish original estimation
            self.transform_points(self.object_points,rotations,translations)
            pc_msg = create_pointcloud_msg('zed2_left_camera_frame', self.new_points)
            self.pub.publish(pc_msg)

            if self.args.depth == True:
                #publish filtered points around original estimation with a distance threshold
                self.filtered_points = sphere_points_around_object(self.depth_map_points,self.t, 0.25)
                pc_msg_icp_depth = create_pointcloud_msg('zed2_left_camera_frame', self.filtered_points)
                self.pub_depth_map_icp.publish(pc_msg_icp_depth)

            if self.args.icp == True:
                # publish original estimation with icp
                self.new_points_icp = icp_pointclouds(self.new_points, self.filtered_points, self.object_points, self.transformation_matrix)
                pc_msg_icp = create_pointcloud_msg('zed2_left_camera_frame',self.new_points_icp)
                self.pub_icp.publish(pc_msg_icp)

    def transform_points(self, points,cam_R_m2c,cam_t_m2c):

        R_transform = Rot.from_euler('x', np.deg2rad(-90)).as_matrix()  
        R = Rot.from_rotvec(cam_R_m2c[0]).as_matrix()
  
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = R
        self.transformation_matrix[3, 3] = 1
        #self.t = np.array([cam_t_m2c[0][2], -cam_t_m2c[0][0],-cam_t_m2c[0][1]],dtype=np.float64) #/2 # t
        self.t = np.array([cam_t_m2c[0][0], cam_t_m2c[0][1],cam_t_m2c[0][2]],dtype=np.float64)/2

        self.transformation_matrix[:3, 3] = self.t
        self.new_points = np.dot(R, self.object_points.T).T + self.t

    def create_pc_from_depth_image(self, data):
        cv_image_depth_raw = self.bridge.imgmsg_to_cv2(data, "32FC1")
        # Get a pointer to the depth values casting the data pointer to floating point
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(np.asarray(cv_image_depth_raw)),self.camera_intrinsic)
        depth_map_points = np.asarray(pcd.points)

        return depth_map_points 


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='arguments for publishing pointclouds')
        parser.add_argument('--icp', default=True, help='Set this flag to enable icp with depth')
        parser.add_argument('--depth', default=True, help='Set this flag to enable depth map to point cloud')

        args = parser.parse_args()

        pc_modifier = PointCloudModifier(args)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass