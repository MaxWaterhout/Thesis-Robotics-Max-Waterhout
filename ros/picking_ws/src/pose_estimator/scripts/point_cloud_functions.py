#!/home/max/env38/bin/python

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

def split_tuple_into_arrays(my_tuple, array_length):
    arrays = [np.array(my_tuple[i:i+array_length]) for i in range(0, len(my_tuple), array_length)]
    return arrays

def create_pointcloud_msg( frame_id, points):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id  # Replace with the actual frame ID of your camera

    # Create the fields for the PointCloud2 message
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),

    ]

    # Create the PointCloud2 message
    modified_pc_msg = pc2.create_cloud(header, fields, points)

    return modified_pc_msg

def icp_pointclouds( source_cloud_points, target_cloud_points, object_points, transformation_matrix):
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_cloud_points)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_cloud_points)

    threshold =0.2
    reg_p2p = o3d.pipelines.registration.registration_icp(source_cloud, target_cloud, threshold, np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint()) 

    matrix = reg_p2p.transformation @ transformation_matrix
    t = matrix[:3,3:].flatten()
    R = matrix[:3,:3]

    new_points_icp = np.dot(R, object_points.T).T + t
    return new_points_icp

            
def sphere_points_around_object(depth_map_points, t, distance_threshold):
    distances = np.linalg.norm(depth_map_points - t, axis=1)
    # Create a boolean mask to filter points within the distance of 0.15
    mask = distances <= distance_threshold

    # Filter the points using the mask
    filtered_points = depth_map_points[mask]

    return filtered_points