#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
from functions_prediction import get_linemod_3d_bboxes
from efficientpose_test.msg import DetectionInfo
from std_msgs.msg import Header
import trimesh
import open3d as o3d
import tf2_ros
from functions_visual import create_visual, split_tuple_into_arrays



class Image_visual:
    def __init__(self, class_to_3d_bboxes):
        #self.graph = tf.Graph()
        #with self.graph.as_default():
            # Initialize your EfficientPose model and load weights
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.bridge = CvBridge()
        self.predict_pub_image = rospy.Publisher("image_prediction_visual", Image)

        self.mesh = trimesh.load(f'/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000001.ply')
        self.model_points = np.array(self.mesh.vertices, dtype=np.float32)  
        self.score_threshold = 0.5
        self.K_matrix = None
        self.distort_list = None
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.camera_frame_id = None

        self.class_to_3d_bboxes = class_to_3d_bboxes
        self.predictions = rospy.Subscriber("image_predictions",DetectionInfo,self.prediction_callback )


    def camera_callback(self, data):
            # This function will be called when a message is received on the topic
            # You can access the camera matrix and other information from the data argument
            if self.K_matrix is None:
                self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
                self.camera_frame_id = data.header.frame_id
                print("The k matrix: ", self.K_matrix)
                print("\n The camera frame id:" , self.camera_frame_id)

    def prediction_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data.predicted_image, "bgr8")
        rotations = split_tuple_into_arrays(data.rotations,3)
        boxes = split_tuple_into_arrays(data.boxes,4) 
        scores = data.scores
        translations = split_tuple_into_arrays(data.translations,3)
        image = create_visual(cv_image, rotations, boxes, scores, translations,self.class_to_3d_bboxes, self.K_matrix)
        self.predict_pub_image.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))        

if __name__ == '__main__':
    try:
        rospy.init_node("visual_node",anonymous = True)
        name_to_3d_bboxes = get_linemod_3d_bboxes()
        class_to_name = {0: "ape"} 
        class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
        detector = Image_visual(class_to_3d_bboxes)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

