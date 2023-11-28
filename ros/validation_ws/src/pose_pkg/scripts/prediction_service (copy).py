#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import tf
import os
import cv2
from functions_prediction import preprocess,postprocess, init_model, build_model
import time
from panda_msgs.msg import DetectionInfo
from std_msgs.msg import Header
import argparse
from scipy.spatial.transform import Rotation
import pandas as pd

class chess_detector:
    def __init__(self,class_to_3d_bboxes, args):
        self.bridge = CvBridge()
        self.K_matrix = None
        self.depth_data = None
        self.image_bbox_pub = rospy.Publisher("/image_bbox", Image)
        self.class_to_3d_bboxes = class_to_3d_bboxes
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.predict_sub = rospy.Subscriber("/image_predictions", DetectionInfo, self.detection_callback)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            print("The k matrix: ", self.K_matrix)

    def detection_callback(self,data):
        if len(data.scores) != 0:
            rotation, translation, df = self.result_to_pandas(data.rotations,data.translations,data.boxes,data.scores,data.name)
            self.publish_tf(rotation,translation,df)
            self.image_bbox_pub.publish(data.predicted_image_bbox)
            

    def publish_tf(self,rotation,translation,df):
        rotation = Rotation.from_rotvec(rotation)
        quaternion = rotation.as_quat()
        self.br.sendTransform(translation, quaternion, rospy.Time.now(), "chess_piece", "zed2_left_camera_optical_frame")     
        
        self.listener.waitForTransform('/charuco','/chess_piece',rospy.Time(), rospy.Duration(4.0))
        (translation_world, rotation_world) = self.listener.lookupTransform('/chess_piece', '/charuco', rospy.Time(0))
        print(translation_world)
        
    def result_to_pandas(self, euler_angles,translation_tuples,boxes,scores,name):

        num_translations = len(translation_tuples) // 3

        # Create lists to hold the split translations
        translation_x = []
        translation_y = []
        translation_z = []

        for i in range(num_translations):
            index = i * 3
            translation_x.append(translation_tuples[index])
            translation_y.append(translation_tuples[index + 1])
            translation_z.append(translation_tuples[index + 2])

        data_dict = {
            'Rot 1': [euler_angles[i] for i in range(0, len(euler_angles), 3)],
            'Rot 2': [euler_angles[i + 1] for i in range(0, len(euler_angles), 3)],
            'Rot 3': [euler_angles[i + 2] for i in range(0, len(euler_angles), 3)],
            'Trans X': translation_x,
            'Trans Y': translation_y,
            'Trans Z': translation_z,
            'Box X1': [boxes[i] for i in range(0, len(boxes), 4)],
            'Box X2': [boxes[i + 1] for i in range(0, len(boxes), 4)],
            'Box Y1': [boxes[i + 2] for i in range(0, len(boxes), 4)],
            'Box Y2': [boxes[i + 3] for i in range(0, len(boxes), 4)],
            'Score_6D_pose':scores,
            'Name': name
            
        }

        # Create a Pandas DataFrame from the dictionary
        df = pd.DataFrame(data_dict)
        print(df)
        # Sort the DataFrame by 'Score' in descending order
        df = df.sort_values(by='Score_6D_pose', ascending=False)


        # Remove rows where x2 or y2 are zero
        df = df[(df['Box X2'] != 0) & (df['Box Y2'] != 0)]
        df = df[(df['Box X1'] != 0) & (df['Box Y1'] != 0)]

        # Reset the index
        df = df.reset_index(drop=True)
    
        df = df.iloc[[0]]

        # Get the rotation and translation of the top row
        top_rotation = df.loc[0, ['Rot 1', 'Rot 2', 'Rot 3']].values
        top_translation = df.loc[0, ['Trans X', 'Trans Y', 'Trans Z']].values
        
        return top_rotation, top_translation,df
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='arguments for service')
        parser.add_argument('--weights', default="/home/max/Documents/ros_workspaces/zed_ws/src/efficientpose_test/scripts/horse/phi_0_linemod_best_ADD.h5", help='path for weights')
        parser.add_argument('--object', default="horse", help='path for weights')
        
        args = parser.parse_args()
        rospy.init_node("detection_service",anonymous = True)
        class_to_3d_bboxes = init_model(args.object)

        detector = chess_detector(class_to_3d_bboxes,args)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

