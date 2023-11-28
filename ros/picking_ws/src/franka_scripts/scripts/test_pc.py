#!/usr/bin/env python

#!/usr/bin/env python
import sys
sys.path.append('/usr/lib/python3/dist-packages/')
import rospy
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import pcl
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class PC_transform:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(5)
        self.sub = rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered", PointCloud2, self.cloud_callback)

        self.pub = rospy.Publisher("/point_cloud/cloud_transformed", PointCloud2, queue_size=1)
        self.rate = rospy.Rate(30.0)
        while not rospy.is_shutdown():
            self.rate.sleep()

    def cloud_callback(self, data):
        
        data.header.frame_id = 'CamLeft'

        self.pub.publish(data)

if __name__ == "__main__":
    rospy.init_node("point_cloud_transform")
    rospy.loginfo("Node started")
    PC = PC_transform()

