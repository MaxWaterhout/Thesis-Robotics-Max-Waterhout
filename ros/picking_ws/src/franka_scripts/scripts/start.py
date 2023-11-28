#!/home/max/env38/bin/python

import rospy
import sys
sys.path.append('/usr/lib/python3/dist-packages/')

import actionlib
from panda_msgs.msg import StartPickAndPlaceAction, StartPickAndPlaceGoal
from panda_msgs.msg import GetPosePredictionsAction, GetPosePredictionsGoal
from move_arm_functions import MovePandaArm
from tf2_geometry_msgs import PoseStamped
from scipy.spatial.transform import Rotation
import tf
from point_cloud_functions import create_pointcloud_msg,sphere_points_around_object, icp_pointclouds
import numpy as np
import trimesh
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import argparse
import pandas as pd

class StartScript:
    def __init__(self):
        self.retry = 0
        parser = argparse.ArgumentParser(description='Script for robotic operations')
        parser.add_argument('--icp', action='store_true', help='Specify if working with a real robot (default is simulated)')
        args = parser.parse_args()
        self.icp = args.icp
        print(f"with icp: {args.icp}")
        rospy.init_node('start_node')
        rospy.loginfo('start.py, waiting for all services to be online..')
        self.move_group_panda = MovePandaArm(homing=True)
        #self.move_group_panda.move_to_predifined_position(self.move_group_panda.start_pos)
        self.move_group_panda.move_to_predifined_position(self.move_group_panda.second_pos)

        self.camera_intrinsic = None
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.depth_data = None
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.bridge = CvBridge()

        self.t = None
        self.transformation_matrix = None
        self.new_points = None
        self.new_points_cam = None
        self.pub = rospy.Publisher('/modified_pointcloud_topic', PointCloud2, queue_size=10)

        self.pub_depth_map = rospy.Publisher('/Pointcloud_Publisher_depth_map', PointCloud2,queue_size=10)
        self.depth_map_points = None
        self.pub_depth_map_icp = rospy.Publisher('/Pointcloud_Publisher_depth_map_icp', PointCloud2,queue_size=10)
        self.filtered_points = None
        
        self.pub_icp = rospy.Publisher('/modified_pointcloud_topic_icp', PointCloud2, queue_size=1)
        self.new_points_icp = None

        rospy.wait_for_service("predictions_horse")

        self.br = tf.TransformBroadcaster()
        self.br_2 = tf.TransformBroadcaster()

        self.listener = tf.TransformListener()
        
        self.mesh_rook = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000001.ply')
        self.mesh_queen = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000002.ply')
        self.mesh_pawn = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000003.ply')
        self.mesh_king = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000004.ply')
        self.mesh_horse = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000005.ply')
        self.mesh_bishop = trimesh.load('/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000006.ply')

        self.object_points = {
            "rook": np.array(self.mesh_rook.vertices, dtype=np.float32) / 1000,
            "queen": np.array(self.mesh_queen.vertices, dtype=np.float32) / 1000,
            "pawn": np.array(self.mesh_pawn.vertices, dtype=np.float32) / 1000,
            "king": np.array(self.mesh_king.vertices, dtype=np.float32) / 1000,
            "horse": np.array(self.mesh_horse.vertices, dtype=np.float32) / 1000,
            "bishop": np.array(self.mesh_bishop.vertices, dtype=np.float32) / 1000
        }

        rospy.loginfo('Lets do the predictions!')
        self.action_client_pose_prediction = actionlib.SimpleActionClient('GetPosePrediction_action', GetPosePredictionsAction)
        self.action_client_pose_prediction.wait_for_server()
        rospy.loginfo('GetPosePrediction_action loaded')
        self.action_client_pick_and_place = actionlib.SimpleActionClient('PickAndPlace_action', StartPickAndPlaceAction)
        self.action_client_pick_and_place.wait_for_server()
        rospy.loginfo('PickAndPlace_action loaded')

        self.final_df = pd.DataFrame()

        while not rospy.is_shutdown():
            self.request_prediction()

        self.move_group_panda.move_to_predifined_position(self.move_group_panda.start_pos)
        print(self.final_df.to_string())
        self.final_df.to_clipboard()

    def camera_callback(self,data):
        if self.camera_intrinsic is None:
            K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            camera_frame_id = data.header.frame_id
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                fx=data.K[0], fy=data.K[4],
                                                cx=data.K[2], cy=data.K[5])
            print("The k matrix: ", K_matrix)

    def depth_callback(self,data):
        self.depth_data = data
        

    def create_pc_from_depth_image(self, data):
        cv_image_depth_raw = self.bridge.imgmsg_to_cv2(data, "32FC1")
        # Get a pointer to the depth values casting the data pointer to floating point
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(np.asarray(cv_image_depth_raw)),self.camera_intrinsic)
        depth_map_points = np.asarray(pcd.points)

        return depth_map_points 

    def request_prediction(self):
        print(self.retry)
        goal = GetPosePredictionsGoal()
        # Fill in the goal here
        self.action_client_pose_prediction.send_goal(goal, feedback_cb=self.feedback_callback_pose)
        self.action_client_pose_prediction.wait_for_result()
        result = self.action_client_pose_prediction.get_result()
        if result:
            if result.success and result.translations != ():
                rotation, translation, df = self.result_to_pandas(result.rotations,result.translations,result.boxes,result.scores, result.name, result.yolo_score)
                self.retry = 0
                translation_world, rotation_world, df= self.transform_pose_to_world_frame(rotation,translation,result.name, df)
                rospy.loginfo("Pose prediction was successful! Requesting pick and place action")

                gripping_result = self.request_pick_and_place(translation_world,rotation_world,result.name)
                df['icp'] = self.icp
                df['gripping_result'] = gripping_result

                self.final_df = pd.concat([self.final_df, df], ignore_index=True)
                print(self.final_df.to_string())
                self.final_df.to_clipboard()
            else:
                rospy.loginfo("Yolo prediction failed, trying from another corner")
                self.retry_prediction()

    def retry_prediction(self):
        if self.retry ==1:
            rospy.loginfo("Stopping script because it cannot detect anything")
            self.move_group_panda.move_to_predifined_position(self.move_group_panda.start_pos)
            print(self.final_df.to_string())
            self.final_df.to_clipboard()
            sys.exit(0)
        if self.retry == 0:
            rospy.loginfo("Trying from left corner")
            self.move_group_panda.move_to_predifined_position(self.move_group_panda.third_pos)   
            self.retry = 1
            self.request_prediction()

    def result_to_pandas(self, euler_angles,translation_tuples,boxes,scores,name,yolo_score):

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
            'Score_Yolo':yolo_score,
            'Name': name
            
        }

        # Create a Pandas DataFrame from the dictionary
        df = pd.DataFrame(data_dict)

        # Sort the DataFrame by 'Score' in descending order
        df = df.sort_values(by='Score_6D_pose', ascending=False)


        # Remove rows where x2 or y2 are zero
        df = df[(df['Box X2'] != 0) & (df['Box Y2'] != 0)]

        # Reset the index
        df = df.reset_index(drop=True)
        # Get the rotation and translation of the top row
        top_rotation = df.loc[0, ['Rot 1', 'Rot 2', 'Rot 3']].values
        top_translation = df.loc[0, ['Trans X', 'Trans Y', 'Trans Z']].values

        return top_rotation, top_translation,df


    def request_pick_and_place(self,translation_world,rotation_world, name):
        goal = StartPickAndPlaceGoal()
        goal.translations = translation_world
        goal.rotations = rotation_world
        goal.name = name
        self.action_client_pick_and_place.send_goal(goal, feedback_cb=self.feedback_callback_pick_and_place)
        self.action_client_pick_and_place.wait_for_result()
        result = self.action_client_pick_and_place.get_result()
        rospy.loginfo(result)

        return result

    
    def flip_z_axis(self,rotation_matrix):
        # Define a 3x3 rotation matrix for a 180-degree rotation around the Z-axis
        R_180_deg = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
            ], dtype=np.float64)

        # Multiply the input rotation matrix by the 180-degree rotation matrix
        flipped_rotation_matrix = np.dot(rotation_matrix, R_180_deg)

        return flipped_rotation_matrix

    def transform_pose_to_world_frame(self, rotation_pred,translation_pred,name, df):

        rotation = Rotation.from_rotvec(rotation_pred)
        if name == 'queen' or name =='rook' or name =='king' or name =='bishop' or name=='pawn':
            rotation_matrix = rotation.as_matrix()
            rotation = self.flip_z_axis(rotation_matrix)
            rotation = Rotation.from_matrix(rotation)

        quaternion = rotation.as_quat()

        self.br.sendTransform(translation_pred, quaternion, rospy.Time.now(), "chess_piece", "CamLeft")
        self.listener.waitForTransform('/world','/chess_piece',rospy.Time(), rospy.Duration(4.0))

        (translation_world, rotation_world) = self.listener.lookupTransform('/world', '/chess_piece', rospy.Time(0))
        if name =='queen' or name=='king':
            translation_world[2] = translation_world[2] - 0.065
            self.br.sendTransform(translation_world, rotation_world, rospy.Time.now(), "chess_piece", "world")
            self.listener.waitForTransform('/world','/chess_piece',rospy.Time(), rospy.Duration(4.0))
        if name =='rook':
            translation_world[2] = translation_world[2] - 0.040
            self.br.sendTransform(translation_world, rotation_world, rospy.Time.now(), "chess_piece", "world")
            self.listener.waitForTransform('/world','/chess_piece',rospy.Time(), rospy.Duration(4.0))
        if name =='bishop':
            translation_world[2] = translation_world[2] - 0.050
            self.br.sendTransform(translation_world, rotation_world, rospy.Time.now(), "chess_piece", "world")
            self.listener.waitForTransform('/world','/chess_piece',rospy.Time(), rospy.Duration(4.0))
        if name =='pawn':
            translation_world[2] = translation_world[2] - 0.035
            self.br.sendTransform(translation_world, rotation_world, rospy.Time.now(), "chess_piece", "world")
            self.listener.waitForTransform('/world','/chess_piece',rospy.Time(), rospy.Duration(4.0))

        (translation_cam, rotation_cam) = self.listener.lookupTransform('/CamLeft', '/chess_piece', rospy.Time(0))
        self.transform_points(self.object_points[str(name)],rotation_world,translation_world,rotation_cam,translation_cam)
        
        #publishing pose estimation
        pc_msg = create_pointcloud_msg('world', self.new_points)
        self.pub.publish(pc_msg)

        #publish the depth map to a pointcloud
        self.depth_map_points = self.create_pc_from_depth_image(self.depth_data)
        pc_msg_depth = create_pointcloud_msg('CamLeft',self.depth_map_points)
        self.pub_depth_map.publish(pc_msg_depth)

        #publish filtered points around original estimation with a distance threshold
        self.filtered_points = sphere_points_around_object(self.depth_map_points,translation_pred, 0.15)
        pc_msg_icp_depth = create_pointcloud_msg('CamLeft', self.filtered_points)
        self.pub_depth_map_icp.publish(pc_msg_icp_depth)

        # publish original estimation with icp
        self.new_points_icp, R_icp, t_icp,rmse_icp,correspondence_set = icp_pointclouds(self.new_points_cam, self.filtered_points, self.object_points[str(name)], self.transformation_matrix)
        pc_msg_icp = create_pointcloud_msg('CamLeft',self.new_points_icp)
        self.pub_icp.publish(pc_msg_icp)
        
        rotation_icp = Rotation.from_matrix(R_icp)

        # Convert the rotation to a quaternion
        quaternion_icp = rotation_icp.as_quat()
        self.br.sendTransform(t_icp, quaternion_icp, rospy.Time.now(), "chess_piece_icp", "CamLeft")
   
        #rospy.loginfo(f"{rotation_world} the rotation of the pose" )
        #rospy.loginfo(f"{translation_world} the translation of the pose")
        self.listener.waitForTransform('/world','/chess_piece_icp',rospy.Time(), rospy.Duration(4.0))

        (translation_world_icp, rotation_world_icp) = self.listener.lookupTransform('/world', '/chess_piece_icp', rospy.Time(0))

        #rospy.loginfo(f"{rotation_world_icp} The rotation of the icp world")
        #rospy.loginfo(f"{translation_world_icp} the translation of the icp world")
        df['Trans X icp'] = translation_world_icp[0]
        df['Trans Y icp'] = translation_world_icp[1]
        df['Trans Z icp'] = translation_world_icp[2]
        df['RMSE icp'] = rmse_icp
        df['Trans X'] = translation_world[0]
        df['Trans Y'] = translation_world[1]
        df['Trans Z'] = translation_world[2]

        columns_to_drop = ['Box X1', 'Box X2', 'Box Y1', 'Box Y2']

        # Use the DataFrame.drop() method to remove the specified columns
        df.drop(columns=columns_to_drop, inplace=True)

        if self.icp:
            return translation_world_icp, rotation_world_icp, df
        else:
            return translation_world, rotation_world, df


    def transform_points(self, points,rotation,translation, rotation_cam, translation_cam):
        rotation = Rotation.from_quat(rotation).as_matrix()
        self.t = np.array([translation[0], translation[1],translation[2]],dtype=np.float64)
        self.new_points = np.dot(rotation, points.T).T + self.t

        rotation = Rotation.from_quat(rotation_cam).as_matrix()

        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = rotation
        self.transformation_matrix[3, 3] = 1
        t = np.array([translation_cam[0], translation_cam[1],translation_cam[2]],dtype=np.float64)
        self.transformation_matrix[:3, 3] = t
        self.new_points_cam = np.dot(rotation, points.T).T + t


    def feedback_callback_pose(self, feedback):
        rospy.loginfo(f"intermediate feedback: any object detections : {feedback.object_detection_succeed}" )

    def feedback_callback_pick_and_place(self, feedback):
        rospy.loginfo(f"test {feedback}" )

def main():
    start_script = StartScript()

if __name__ == '__main__':
    main()
