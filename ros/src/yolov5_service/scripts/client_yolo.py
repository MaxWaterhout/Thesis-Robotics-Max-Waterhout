#!/home/max/env38/bin/python

from gc import callbacks
from queue import Empty
import re
import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np

from std_msgs.msg import Empty, String
from panda_msgs.srv import InputYoloPrediction, InputYoloPredictionResponse
from panda_msgs.srv import InputPrediction,InputPredictionResponse
import actionlib
from panda_msgs.msg import GetPosePredictionsAction, GetPosePredictionsFeedback, GetPosePredictionsResult


class yolo_client:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_data = None
        self.K_matrix = None
        self.depth_data = None
        
        #self.prediction_service = rospy.Service('chess_prediction', predictionImage, self.handle_prediction_request)
        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)

        self.trigger_sub = rospy.Subscriber("yolo_prediction_trigger", Empty, self.trigger_callback)

        self.pose_publisher = rospy.Publisher("6D_prediction_visual", Image)

        self.action_server = actionlib.SimpleActionServer("GetPosePrediction_action",GetPosePredictionsAction,self.execute_cb,auto_start=False)
        self.action_server.start()
        self.feedback = GetPosePredictionsFeedback()
        self.result = GetPosePredictionsResult()
        rospy.loginfo("Loaded yolo client")

    def depth_callback(self,data):
        self.depth_data = data
    
    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            rospy.loginfo(f"Loaded the K matrix")

    def image_callback(self, data):
        self.image_data = data

    def trigger_callback(self,msg):
        rospy.loginfo("Getting a prediction request from /yolo_prediction_trigger")
        action = False
        response = self.handle_prediction_request()            

    def execute_cb(self, msg):
        rospy.loginfo("Getting a prediction request from /yolo_prediction_trigger")
        response, name = self.handle_prediction_request(action=True)
        if response:
            rospy.loginfo(f"The rotations of {name} is {response.rotations}, the translations: {response.translations} with confidence: {response.scores}")
            self.result.success = True  # Set success to True
            self.result.rotations = response.rotations
            self.result.translations = response.translations
            self.result.name = name
            self.action_server.set_succeeded(self.result)
        else:
            rospy.loginfo('No detection between yolo or efficientpose')
            self.action_server.set_aborted()

    def handle_prediction_request(self,action=False):
        rospy.wait_for_service("prediction_yolo")  # Wait for the service to become available
        try:
            prediction_service = rospy.ServiceProxy("prediction_yolo", InputYoloPrediction)
            # Create a request object
            request = InputYoloPrediction._request_class()
            request.predicted_image = self.image_data
            
            # Call the service with the request
            response = prediction_service(request)

            if response.name.data and response.score > 0.8:
                self.feedback.object_detection_succeed = True
                if action:
                    self.action_server.publish_feedback(self.feedback)
                name = response.name.data
                response = self.handle_6D_prediction_request(response.name.data)
            else:
                self.feedback.object_detection_succeed = False
                if action:
                    self.action_server.publish_feedback(self.feedback)
                response = None
                name = None
            return response, name
            
        except rospy.ServiceException as e:
            print("Service call failed:", str(e))

    def handle_6D_prediction_request(self, name):
        service_name = "predictions_{}".format(name)
        rospy.wait_for_service(service_name)  # Wait for the service to become available
        try:
            prediction_service = rospy.ServiceProxy(service_name, InputPrediction)
            # Create a request object
            request = InputPrediction._request_class()
            request.predicted_image = self.image_data
            request.depth_map = self.depth_data
            request.K_matrix = self.K_matrix.flatten()
            # Call the service with the request
            response = prediction_service(request)
            self.pose_publisher.publish(response.predicted_image_bbox)
            return response
        
        except rospy.ServiceException as e:
                print("Service call failed:", str(e))

if __name__ == '__main__':
    try:
        rospy.init_node("yolo_detection_node_client",anonymous = True)
        detector = yolo_client()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

