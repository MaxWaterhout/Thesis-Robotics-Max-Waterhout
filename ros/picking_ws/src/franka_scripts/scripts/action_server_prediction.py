#!/home/max/env38/bin/python

import rospy
import actionlib
from franka_scripts import GetPosePredictionAction, GetPosePredictionFeedback, GetPosePredictionResult

import sys
import moveit_commander
import moveit_msgs.msg


class ActionServer():
    def __init__(self):
        self.action_server = actionlib.SimpleActionServer("GetPosePrediction_action",GetPosePredictionAction,self.execute_cb,auto_start=True)

    def execute_cb(self, goal):
        succes = True
        feedback = GetPosePredictionFeedback()
        result = GetPosePredictionResult()

        if self.action_server.is_preempt_requested():
            print(goal)
            succes = False
        
if __name__ == '__main__':
    rospy.init_node('action_server_prediction')
    server = ActionServer()
    rospy.spin()
