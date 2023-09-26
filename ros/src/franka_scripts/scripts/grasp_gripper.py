#!/usr/bin/env python

import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

if __name__ == '__main__':
    try:
        # Initialize the ROS node
        rospy.init_node('gripper_control_node')
        
        # Create an action client for the GripperCommand action
        client = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)
        
        # Wait for the action server to start
        client.wait_for_server()
        
        # Create a GripperCommandGoal with desired width and max_effort
        goal = GripperCommandGoal()
        goal.command.position = 0.01  # Desired width (adjust as needed)
        goal.command.max_effort = 0.0  # Max effort (adjust as needed)
        
        # Send the goal to the action server
        client.send_goal(goal)
        
        # Wait for the gripper action to complete (you can set a timeout here)
        client.wait_for_result()
        
        # Check the result of the gripper action (optional)
        result = client.get_result()
        if result:
            rospy.loginfo('Gripper action completed successfully')
        else:
            rospy.logwarn('Gripper action did not succeed')


        rospy.sleep(3)

        goal = GripperCommandGoal()
        goal.command.position = 0.02  # Desired width (adjust as needed)
        goal.command.max_effort = 0  # Max effort (adjust as needed)
        
        # Send the goal to the action server
        client.send_goal(goal)
        
        # Wait for the gripper action to complete (you can set a timeout here)
        client.wait_for_result()
        
        # Check the result of the gripper action (optional)
        result = client.get_result()
    
    except rospy.ROSInterruptException:
        rospy.logerr('ROS node interrupted')
