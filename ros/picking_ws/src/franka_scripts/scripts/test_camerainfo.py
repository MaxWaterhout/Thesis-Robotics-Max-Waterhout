#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_callback(data):
    # Modify the frame_id
    data.header.frame_id = "LeftCam"

    # Publish the modified CameraInfo message
    pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('camera_info_republisher', anonymous=True)

    # Create a subscriber to the original topic
    rospy.Subscriber("/zed2/zed_node/left/camera_info", CameraInfo, camera_info_callback)

    # Create a publisher with the new frame_id
    pub = rospy.Publisher("/zed2/zed_node/left/camera_info_modified", CameraInfo, queue_size=10)

    rospy.spin()