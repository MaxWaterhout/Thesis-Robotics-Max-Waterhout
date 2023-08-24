#!/home/max/env38/bin/python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from math import pi, tau, dist, fabs, cos

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def move_to_start_position():
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)    
    display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path",moveit_msgs.msg.DisplayTrajectory,queue_size=20)
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = -0.0002861509839998449
    joint_goal[1] = -1.5649661966610398
    joint_goal[2] = 0.001571490184083701
    joint_goal[3] = -2.441184105685472
    joint_goal[4] = 0.009147199263040286
    joint_goal[5] = 1.397496153041122
    joint_goal[6] = 0.785220506046326

    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()
    print("")