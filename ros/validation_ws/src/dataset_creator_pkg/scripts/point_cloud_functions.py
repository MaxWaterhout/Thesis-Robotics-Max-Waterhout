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

    threshold =0.15
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=15,  # Maximum number of iterations
        relative_fitness=0.0001  # Maximum transformation error for convergence
        )
    reg_p2p = o3d.pipelines.registration.registration_icp(source_cloud, target_cloud, threshold, np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria=criteria) 
    matrix = reg_p2p.transformation @ transformation_matrix
    t = matrix[:3,3:].flatten()
    R = matrix[:3,:3]

    new_points_icp = np.dot(R, object_points.T).T + t
    return new_points_icp ,R, t, reg_p2p.inlier_rmse, reg_p2p.correspondence_set

            
def sphere_points_around_object(depth_map_points, t, distance_threshold):
    # Check for NaN values
    nan_indices = np.isnan(depth_map_points)
    mask = ~np.all(nan_indices, axis=1)

    # Use the mask to filter the array and remove rows with all False values
    filtered_array = depth_map_points[mask].astype(float)
    test = filtered_array - t
    t = t.astype(float)

    depth_map_points = depth_map_points.astype(float)
    distances = np.linalg.norm(depth_map_points - t, axis=1)
    # Create a boolean mask to filter points within the distance of 0.15
    mask = distances <= distance_threshold

    # Filter the points using the mask
    filtered_points = depth_map_points[mask]
    point_cloud_within_threshold = o3d.geometry.PointCloud()
    point_cloud_within_threshold.points = o3d.utility.Vector3dVector(filtered_points)

    plane_model, inliers = point_cloud_within_threshold.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    filtered_points = point_cloud_within_threshold.select_by_index(inliers, invert=True)
    filtered_points = np.asarray(filtered_points.points)

    return filtered_points