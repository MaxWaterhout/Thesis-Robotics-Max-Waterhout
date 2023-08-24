#!/home/max/env38/bin/python

import rospy
import actionlib
from franka_scripts import StartPickAndPlaceAction, StartPickAndPlaceGoal

class ActionClient():
    def __init__(self):
        self.action_client = actionlib.SimpleActionClient('StartPickAndPlace_client', StartPickAndPlaceAction)
        self.action_client.wait_for_server()

        goal = StartPickAndPlaceGoal()
        # Fill in the goal here
        self.action_client.send_goal(goal)
        self.action_client.wait_for_result(rospy.Duration.from_sec(5.0))
        

if __name__ == '__main__':
    rospy.init_node('action_client_arm')
    client = ActionClient()
    rospy.spin()
