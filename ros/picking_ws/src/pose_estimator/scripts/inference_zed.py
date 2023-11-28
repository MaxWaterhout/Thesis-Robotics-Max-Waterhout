#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from model import build_EfficientPose
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
import os
import cv2
from functions_prediction import get_linemod_3d_bboxes, preprocess,postprocess, project_bbox_3D_to_2D, draw_bbox_8_2D
from utils import preprocess_image
import time

class chess_detector:
    def __init__(self,num_classes,class_to_3d_bboxes):
        #self.graph = tf.Graph()
        #with self.graph.as_default():
            # Initialize your EfficientPose model and load weights
        self.model, self.image_size = self.build_model(num_classes)

        self.session = tf.compat.v1.keras.backend.get_session()

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("image_detection", Image)
        self.phi = 0
        self.score_threshold = 0.5
        self.K_matrix = None
        self.class_to_3d_bboxes = class_to_3d_bboxes
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)

    def build_model(self,num_classes):
        print("\nBuilding model...\n")
        _, model, _ = build_EfficientPose(0,
                                    num_classes = num_classes,
                                    num_anchors = 9,
                                    freeze_bn = True,
                                    score_threshold = 0.5,
                                    num_rotation_parameters = 3,
                                    print_architecture = False)
        print("\n Build model!")
        print("\n loading weights")
        model.load_weights("/home/max/Documents/ros_workspaces/zed_ws/src/efficientpose_test/scripts/phi_0_linemod_best_ADD-S.h5", by_name=True)
        print("Done!")
        image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
        image_size = image_sizes[0]

        return model, image_size
    
    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            print("The k matrix: ", self.K_matrix)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = self.prediction(cv_image)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


    def prediction(self,image):
        score_threshold = 0.5
        #preprocessingd
        #predict
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)
            input_list, scale = preprocess(image, self.image_size, self.K_matrix, 1)
            boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
            boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)

            if len(boxes) != 0:
                for i in range(len(boxes)):
                    if scores[i] > 0.1:

                        x1, y1, x2, y2 = [boxes[i][0],boxes[i][1], boxes[i][2], boxes[i][3]]

                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))

                        # Draw the rectangle on the image
                        cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

            
            if len(labels) != 0:
                points_bbox_2D = project_bbox_3D_to_2D(self.class_to_3d_bboxes[labels[0]], rotations[0, :], translations[0], self.K_matrix, append_centerpoint = True)
                draw_bbox_8_2D(image, points_bbox_2D)

                print(self.class_to_3d_bboxes[0])
            else:
                None

        return image
    
def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)


def init_model():
    """
    Run EfficientPose in inference mode live on webcam.

    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()
    
    # save_path = "./predictions/occlusion/" #where to save the images or None if the images should be displayed and not saved
    save_path = None
    image_extension = ".jpg"
    class_to_name = {0: "ape"} #Occlusion
    #class_to_name = {0: "driller"} #Linemod use a single class with a name of the Linemod objects
    translation_scale_norm = 1000.0
    draw_bbox_2d = True
    draw_name = False
    num_classes = len(class_to_name)

    #you probably need to replace the linemod camera matrix with the one of your webcam
    name_to_3d_bboxes = get_linemod_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    return num_classes, class_to_3d_bboxes







        

if __name__ == '__main__':
    try:
        rospy.init_node("detection_node",anonymous = True)
        num_classes, class_to_3d_bboxes = init_model()
        detector = chess_detector(num_classes, class_to_3d_bboxes)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
        #phi, num_classes, score_threshold, path_to_weights = init_model()
        #model = build_model(phi, num_classes, score_threshold)
        #model, image_size = loading_weights(model, path_to_weights, phi)


    except rospy.ROSInterruptException:
        pass
