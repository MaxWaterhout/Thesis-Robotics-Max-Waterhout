#!/home/max/env38/bin/python

import roslib
import math
import rospy
import tf
import tf2_ros
import geometry_msgs.msg as gm

if __name__ == '__main__':
    rospy.init_node('fixed_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    translation = (0.0581, -0.0636, 0.0319)
    # Rotation quaternion (x, y, z, w) for 90-degree rotation around Z-axis
    angle = math.pi / 2.0  # 90 degrees in radians
    rotation = (0.00565, 0.0050, 0.696, 0.718)
    #rotation = (0,0,0,1)
    # Broadcast the transform
    rospy.loginfo("Added camleft to tf tree")

    while not rospy.is_shutdown():
        br.sendTransform(translation, rotation, rospy.Time.now(), "CamLeft", "fr3_hand")
        rate.sleep()

    rospy.loginfo("stopped scripts")
