#!/home/max/env38/bin/python

import rospy
import actionlib
from franka_scripts import StartPickAndPlaceAction, StartPickAndPlaceFeedback, StartPickAndPlaceResult

import sys
import moveit_commander
import moveit_msgs.msg


class ActionServer():
    def __init__(self):
        self.action_server = actionlib.SimpleActionServer("StartPickAndPlace_action",StartPickAndPlaceAction,self.execute_cb,auto_start=True)


    def execute_cb(self, goal):
        succes = True
        feedback = StartPickAndPlaceFeedback()
        result = StartPickAndPlaceResult()

        if self.action_server.is_preempt_requested():
            print(goal)
            succes = False
        self.action_server.set_succeeded()
        

if __name__ == '__main__':
    rospy.init_node('action_server_arm')
    server = ActionServer()
    rospy.spin()
