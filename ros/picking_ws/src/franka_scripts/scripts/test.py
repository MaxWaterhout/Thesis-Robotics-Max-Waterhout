#!/home/max/env38/bin/python
import rospy
import sys
import tf
import time
import rospy
import sys
sys.path.append('/usr/lib/python3/dist-packages/')
from move_arm_functions import MovePandaArm
import tf
from geometry_msgs.msg import Pose

import tf2_ros
import tf2_geometry_msgs 
rospy.init_node('test')

move_group_panda = MovePandaArm()
move_group_panda.move_to_predifined_position(move_group_panda.start_pos)
#move_group_panda.move_to_predifined_position(move_group_panda.second_pos)
#move_group_panda.move_to_predifined_position(move_group_panda.third_pos)
"""
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
rospy.sleep(2.0)

pose = Pose()
pose.position.x = 0
pose.position.y = 0
pose.position.z = 0.05

pose_stamped = tf2_geometry_msgs.PoseStamped()
pose_stamped.pose = pose
pose_stamped.header.frame_id = 'fr3_hand_tcp'
pose_stamped.header.stamp = rospy.Time.now()

# ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
output_pose_stamped = tf_buffer.transform(pose_stamped, 'world', rospy.Duration(1))
   
print(output_pose_stamped)
"""
rospy.sleep(2)

#move_group_panda.move_ee_forward([0,0,-0.1])