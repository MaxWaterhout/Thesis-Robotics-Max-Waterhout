#!/home/max/env38/bin/python

import rospy
import actionlib
from move_arm_functions import MovePandaArm
import sys
import argparse
sys.path.append('/usr/lib/python3/dist-packages/')

class ActionClient():
    def __init__(self):
        rospy.init_node("action_client_arm",anonymous = True)
        self.move_group_panda = MovePandaArm()
        self.move_group_panda.reach_pose()

if __name__ == '__main__':
    rospy.loginfo("starting action client arm")
    client = ActionClient()
    rospy.spin()
