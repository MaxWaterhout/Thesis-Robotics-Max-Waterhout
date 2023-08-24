#!/home/max/env38/bin/python

import rospy
import actionlib
from panda_msgs.msg import StartPickAndPlaceAction, StartPickAndPlaceGoal
from panda_msgs.msg import GetPosePredictionsAction, GetPosePredictionsGoal
from move_arm_functions import move_to_start_position

class StartScript:
    def __init__(self):
        rospy.init_node('start_node')
        #move_to_start_position()
        self.action_client = actionlib.SimpleActionClient('GetPosePrediction_action', GetPosePredictionsAction)
        self.action_client.wait_for_server()

        goal = GetPosePredictionsGoal()
        # Fill in the goal here
        self.action_client.send_goal(goal, feedback_cb=self.feedback_callback)
        self.action_client.wait_for_result()
        result = self.action_client.get_result()
        if result:
            if result.success:
                rospy.loginfo("Action was successful!")
                rospy.loginfo("Rotations: {}".format(result.rotations))
                rospy.loginfo("Translations: {}".format(result.translations))
                rospy.loginfo("Name: {}".format(result.name))
            else:
                rospy.loginfo("Action failed.")
        rospy.sleep(1)

    def feedback_callback(self, feedback):
        rospy.loginfo(f"intermediate feedback: any object detections : {feedback.object_detection_succeed}" )

def main():
    start_script = StartScript()
    #start_script.send_pick_and_place_goal()

if __name__ == '__main__':
    main()
