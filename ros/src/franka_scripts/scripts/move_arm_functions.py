#!/home/max/env38/bin/python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import math 
sys.path.append('/usr/lib/python3/dist-packages/')

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs import msg
from scipy.spatial.transform import Rotation
import numpy as np
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from franka_gripper.msg import HomingAction, HomingGoal
import tf2_ros
import tf2_geometry_msgs 
#PICK_ORIENTATION_EULER = [-math.pi, 0, -math.pi / 2]
PICK_ORIENTATION_EULER = [-math.pi, 0, 0]

(X, Y, Z, W) = (0, 1, 2, 3)


class MovePandaArm:
    def __init__(self,real_robot=True,homing=False):
        self.real_robot = real_robot
        if self.real_robot:
            self.client = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)
            # Wait for the action server to start
            self.client.wait_for_server()
            self.client_homing = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
    
            # Wait for the action server to become available
            self.client_homing.wait_for_server()
            
            # Create a HomingGoal and send it to the action server
            print(homing, 'homing')
            if homing:
                goal = HomingGoal()
                self.client_homing.send_goal(goal)
                
                # Wait for the gripper to finish homing
                self.client_homing.wait_for_result()
            self.gripper_result = None
        print(f"Real_robot: {self.real_robot}")
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node('action_client_arm')
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()        

        self.start_pos =  [0.07386156003279816, -1.4929562226478224, 0.07369458524744465, -2.7279904777741244, 0.04736358604606546, 1.8988232214561627, 0.9123896112901707, 0.04044574126601219, 0.04044574126601219]
        self.second_pos =   [-0.17532954995229752, -1.3193210180289827, -0.9512862434207381, -2.74122596424223, 0.022185408831416117, 1.929669392971922, -0.39047594749863573, 0.03484765812754631, 0.03484765812754631]
        self.third_pos = [-0.2252873845318504, -1.3311063479438248, 1.323365618567643, -2.497673709161204, 0.017002795519390643, 1.9311177374590178, 2.379355021035231, 0.04044574126601219, 0.04044574126601219]

        self.chess_pieces = {
            "horse": {"height": 0.045, "width": 0.006},
            "rook": {"height": 0.040, "width": 0.010},
            "pawn": {"height": 0.032, "width": 0.0075},
            "bishop": {"height": 0.040, "width": 0.0075},
            "queen": {"height": 0.055, "width": 0.010},
            "king": {"height": 0.061, "width": 0.010}}
        
        self.add_table()
        self.box_pos = [0.4,0.4,0]
        self.add_box()
        self.add_standing_box()
        self.add_back_wall()
        self.add_side_walls()
        group_name = "fr3_arm"
        self.arm = moveit_commander.MoveGroupCommander(group_name) 
        self.arm.set_end_effector_link("fr3_hand_tcp")
        self.arm.set_planning_time(30)

        group_name = "fr3_hand"
        self.gripper = moveit_commander.MoveGroupCommander("fr3_hand") 

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
    
    def move_to_predifined_position(self, joint_goal_values):
        rospy.loginfo('Moving the arm to a position')
        joint_goal = self.arm.get_current_joint_values()
        joint_goal[0] = joint_goal_values[0]
        joint_goal[1] = joint_goal_values[1]
        joint_goal[2] = joint_goal_values[2]
        joint_goal[3] = joint_goal_values[3]
        joint_goal[4] = joint_goal_values[4]
        joint_goal[5] = joint_goal_values[5]
        joint_goal[6] = joint_goal_values[6]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.arm.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.arm.stop()
        rospy.loginfo('Arm in position')

    def move_gripper(self, width_grip = 0.04):
        if self.real_robot:
            print(f"grabbing with width {width_grip}")
            # Create a GripperCommandGoal with desired width and max_effort
            goal = GripperCommandGoal()
            goal.command.position = width_grip  # Desired width (adjust as needed)
            goal.command.max_effort = 20.0  # Max effort (adjust as needed)
            if width_grip==0.04:
                goal.command.max_effort = 0
            
            # Send the goal to the action server
            self.client.send_goal(goal)
            
            # Wait for the gripper action to complete (you can set a timeout here)
            self.client.wait_for_result()
            
            # Check the result of the gripper action (optional)
            result = self.client.get_result()
            if width_grip != 0.04:
                if result:
                    self.gripper_result = True
                    rospy.loginfo('Gripper action completed successfully')
                else:
                    rospy.logwarn('Gripper action did not succeed')
                    self.gripper_result = False
        else:
            self.gripper.set_joint_value_target([width_grip,width_grip])
            self.gripper.go() 


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
        p.pose.position.y = 0.6
        p.pose.position.z = 0
        self.scene.add_box("side_wall", p, (1.5, 0.01, 1.5))
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0
        p.pose.position.y = -0.6
        p.pose.position.z = 0
        self.scene.add_box("side_wall_2", p, (1.5, 0.01, 1.5))

    def add_table(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0
        self.scene.add_box("table", p, (1.5, 1.5, 0.01))

    def add_box(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = self.box_pos[0]
        p.pose.position.y = self.box_pos[1]
        p.pose.position.z = self.box_pos[2]
        self.scene.add_box("box", p, (0.2, 0.2, 0.05))

    def add_standing_box(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.4
        p.pose.position.y = 0
        p.pose.position.z = 0
        self.scene.add_box("standing box", p, (0.5, 0.5, 0.54))

    def move_ee_in_frame(self,position=[0,0,0.05]):
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = 'fr3_hand_tcp'
        pose_stamped.header.stamp = rospy.Time.now()

        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        pose = self.tf_buffer.transform(pose_stamped, 'world',rospy.Duration(4)).pose

        pose.orientation = self.arm.get_current_pose().pose.orientation

        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.clear_pose_targets()

    def wrap_to_2pi(self,value):
        return (value % (2 * math.pi) + 2 * math.pi) % (2 * math.pi)

    def pick_and_place_object(self,position=[0,0,0],orientation_world=[0,0,0],name=None):

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2] + self.chess_pieces[str(name)]['height']
        print(f"picking at height {self.chess_pieces[str(name)]['height']}")
        orientation_chess = PICK_ORIENTATION_EULER
        # Create a Rotation object from the quaternion
        rotation = Rotation.from_quat(orientation_world)

        # Convert the rotation to Euler angles (roll, pitch, yaw)
        euler_angles = rotation.as_euler('xyz', degrees=True)  # Specify the order and degrees=True for degrees

        if name == 'horse':
            orientation_chess[2] = orientation_chess[2] + np.deg2rad(euler_angles[2]) + 0.5*math.pi
            wrapped_values = [self.wrap_to_2pi(value) for value in orientation_chess]
            orientation_chess[2] = wrapped_values[2]
            print(wrapped_values)
            print(orientation_chess, "orientation horse")
  
        orientation = quaternion_from_euler(*orientation_chess)
        pose.orientation.x = orientation[X]
        pose.orientation.y = orientation[Y]
        pose.orientation.z = orientation[Z]
        pose.orientation.w = orientation[W]
        
        pregrasp_pose = Pose()
        pregrasp_pose.position.x = position[0]
        pregrasp_pose.position.y = position[1]
        pregrasp_pose.position.z = position[2] + self.chess_pieces[str(name)]['height'] + 0.05
        pregrasp_pose.orientation.x = orientation[X]
        pregrasp_pose.orientation.y = orientation[Y]
        pregrasp_pose.orientation.z = orientation[Z]
        pregrasp_pose.orientation.w = orientation[W]
        self.reach_pose(pose,pregrasp_pose,name)

        pose = Pose()
        pose.position.x = self.box_pos[0]
        pose.position.y = self.box_pos[1]
        pose.position.z = 0.43
        orientation = quaternion_from_euler(*PICK_ORIENTATION_EULER)
        pose.orientation.x = orientation[X]
        pose.orientation.y = orientation[Y]
        pose.orientation.z = orientation[Z]
        pose.orientation.w = orientation[W]
        self.place_pose(pose)

        return self.gripper_result

    def reach_pose(self, pose,pregrasp_pose,name, position_tolerance=0.01, orientation_tolerance=0.1):
        self.arm.set_pose_target(pregrasp_pose)
        self.arm.set_goal_position_tolerance(position_tolerance)
        self.arm.set_goal_orientation_tolerance(orientation_tolerance)

        self.arm.go(wait=True)
        self.arm.clear_pose_targets()
        rospy.loginfo('Arm in pregrasp position')
        
        self.arm.set_pose_target(pose)
        self.arm.set_goal_position_tolerance(position_tolerance)
        self.arm.set_goal_orientation_tolerance(orientation_tolerance)

        self.arm.go(wait=True)
        self.arm.clear_pose_targets()
        rospy.loginfo('Arm in final position')
        self.move_gripper(self.chess_pieces[str(name)]['width'])
        self.arm.set_pose_target(pregrasp_pose)
        self.arm.set_goal_position_tolerance(position_tolerance)
        self.arm.set_goal_orientation_tolerance(orientation_tolerance)
        self.arm.go(wait=True)
        self.arm.clear_pose_targets()

    def place_pose(self,pose,tolerance=0.001):
        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.clear_pose_targets()
        pose.position.z = 0.2
        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.clear_pose_targets()
        rospy.loginfo('Arm in placing position')

        self.move_gripper()
        self.move_to_predifined_position(self.second_pos)





