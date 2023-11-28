#!/home/max/env38/bin/python

import rospy
import actionlib
from panda_msgs.msg import StartPickAndPlaceAction, StartPickAndPlaceGoal,StartPickAndPlaceResult,StartPickAndPlaceFeedback
from move_arm_functions import MovePandaArm
import sys
import argparse
sys.path.append('/usr/lib/python3/dist-packages/')

class ActionClient():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script for robotic operations')
        parser.add_argument('--real-robot', action='store_true', help='Specify if working with a real robot (default is simulated)')
        args = parser.parse_args()
        rospy.init_node("action_client_arm",anonymous = True)
        self.result = StartPickAndPlaceResult()
        self.feedback = StartPickAndPlaceFeedback()
        self.move_group_panda = MovePandaArm()

        self.action_client = actionlib.SimpleActionServer("PickAndPlace_action",StartPickAndPlaceAction,self.execute_cb,auto_start=False)
        self.action_client.start()

    def execute_cb(self,msg):
        rospy.loginfo("received a prediction and starting pick and place action")

        gripper_result = self.move_group_panda.pick_and_place_object(msg.translations,msg.rotations,msg.name)

        self.feedback.pick_planning_success = True
        self.feedback.picking_success = True
        self.feedback.place_planning_success = True
        self.feedback.placing_success = True
        self.action_client.publish_feedback(self.feedback)
        self.result.success = gripper_result
        self.action_client.set_succeeded(self.result)

        

if __name__ == '__main__':
    rospy.loginfo("starting action client arm")
    client = ActionClient()
    rospy.spin()
