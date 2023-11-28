#!/home/max/env38/bin/python

import sys
import rospy
import moveit_commander
import math 
sys.path.append('/usr/lib/python3/dist-packages/')

import open3d as o3d
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
import numpy as np
import time 
#PICK_ORIENTATION_EULER = [-math.pi, 0, -math.pi / 2]
PICK_ORIENTATION_EULER = [-math.pi, 0, 0]
from std_msgs.msg import Empty
from std_msgs.msg import Int32


(X, Y, Z, W) = (0, 1, 2, 3)


class MovePandaArm:
    def __init__(self,real_robot=False,homing=False):
        self.real_robot = real_robot
        print(f"Real_robot: {self.real_robot}")
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()        
        self.activation_pub = rospy.Publisher('/activation_topic', Int32, queue_size=1)

        self.add_table()
        self.add_standing_box()
        self.add_back_wall()
        self.add_side_walls()
        self.coordinate_frames = []
        self.rotations = []
        self.create_coord_frames()
        group_name = "fr3_arm"
        self.arm = moveit_commander.MoveGroupCommander(group_name) 
        self.arm.set_end_effector_link("fr3_hand_tcp")
        self.arm.set_planning_time(7)
        self.index = None
        group_name = "fr3_hand"
        self.gripper = moveit_commander.MoveGroupCommander("fr3_hand") 
        start_index = 0
        print("length of coordinate frames:",len(self.coordinate_frames))
        for j in range(len(self.coordinate_frames)):
            self.index = start_index + j 
            coordinate = self.coordinate_frames[self.index]
            print(f"position {self.index} of {len(self.coordinate_frames)} with ")
            position = coordinate.get_center()
            orientation = Rotation.from_matrix(self.rotations[self.index]).as_quat()
            self.reach_pose(position,orientation)

    def add_back_wall(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = -0.45
        p.pose.position.y = 0
        p.pose.position.z = 0
        self.scene.add_box("back_wall", p, (0.01, 1.5, 1.5))

    def add_side_walls(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0
        p.pose.position.y = 0.65
        p.pose.position.z = 0
        self.scene.add_box("side_wall", p, (1.5, 0.01, 1.5))
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0
        p.pose.position.y = -0.7
        p.pose.position.z = 0
        self.scene.add_box("side_wall_2", p, (1.5, 0.01, 1.5))

    def add_table(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0
        self.scene.add_box("table", p, (1.5, 1.5, 0.01))

    def add_standing_box(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.83
        p.pose.position.y = 0
        p.pose.position.z = 0
        self.scene.add_box("standing box", p, (0.5, 0.5, 0.58))

    def reach_pose(self,position,orientation, position_tolerance=0.01, orientation_tolerance=0.1):
        mean = 0.0
        std_dev = 0.05  # Adjust this value based on your desired range

        # Generate random values from a normal distribution
        x_offset = np.random.normal(mean, std_dev)
        y_offset = np.random.normal(mean, std_dev)
        z_offset = np.random.normal(mean, std_dev)


        # Add the random offsets to the position
        new_position = [position[0] + x_offset, position[1] + y_offset, position[2] + z_offset]

        pregrasp_pose = Pose()
        pregrasp_pose.position.x = new_position[0]
        pregrasp_pose.position.y = new_position[1]
        pregrasp_pose.position.z = new_position[2] 
        pregrasp_pose.orientation.x = orientation[0]
        pregrasp_pose.orientation.y = orientation[1]
        pregrasp_pose.orientation.z = orientation[2]
        pregrasp_pose.orientation.w = orientation[3]

        self.arm.set_pose_target(pregrasp_pose)
        self.arm.set_goal_position_tolerance(position_tolerance)
        self.arm.set_goal_orientation_tolerance(orientation_tolerance)

        plan = self.arm.plan()
        if plan[0] ==False:
            rospy.logerr("Motion planning failed or timed out.")
            return

        success = self.arm.execute(plan[1], wait=True)

        if not success:
            rospy.logerr("Execution failed or timed out.")
            # Handle the execution failure here
            return

        self.arm.clear_pose_targets()
        activation_msg = Int32()
        activation_msg.data = self.index

        self.activation_pub.publish(activation_msg)

    def create_coord_frames(self):
        # Define a list of radii to loop through
        radii = [0.3,0.45,0.6]
        num_points = 10

        # Specify the theta and phi angle ranges
        min_theta = np.radians(15)
        max_theta = np.radians(75)
        min_phi = np.radians(120)
        max_phi = np.radians(240)

        # Create an empty point cloud
        point_clouds = []

        for radius in radii:
            # Create an empty point cloud for the current radius
            point_cloud = o3d.geometry.PointCloud()
            
            # Sample points on the surface of a hemisphere
            theta = np.linspace(min_theta, max_theta, num_points)
            phi = np.linspace(min_phi, max_phi, num_points)
            theta, phi = np.meshgrid(theta, phi)

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            # Stack the points as a (3, num_points^2) array
            points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
            mask = points[:, 2] >= 0
            points = points[mask]

            center = np.array([0.83, 0, 0.26])
            translated_points = points + center

            # Set the points in the point cloud
            point_cloud.points = o3d.utility.Vector3dVector(translated_points)
            
            # Append the current point cloud to the list
            point_clouds.append(point_cloud)

        self.frame_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=center)

        self.frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        for point_cloud in point_clouds:
            points = np.asarray(point_cloud.points)
            for point in points:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

                # Calculate the transformation matrix to make the z-axis point to the center
                z_axis = center - point
                z_axis /= np.linalg.norm(z_axis)
                x_axis = np.array([0, 0, 1])
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)

                transformation = np.eye(4)
                transformation[:3, 0] = x_axis
                transformation[:3, 1] = y_axis
                transformation[:3, 2] = z_axis
                transformation[:3, 3] = point
                # Apply the custom transformation to the frame
                frame.transform(transformation)
                self.coordinate_frames.append(frame)
                # Save the rotation part of the transformation
                rotation_matrix = transformation[:3, :3]
                self.rotations.append(rotation_matrix)
        
